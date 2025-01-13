/**
 * @file Utility functions to interact with the Hugging Face Hub (https://huggingface.co/models)
 *
 * @module utils/hub
 */

import fs from "fs";
import path from "path";
import { env } from "./env.js";
import { dispatchCallback } from "./core.js";
import type { PretrainedOptions, CustomCache } from "./interfaces.js";
import {
	type ErrorMapping,
	ERROR_MAPPING,
	CONTENT_TYPE_MAP,
	type ContentTypeMap,
} from "../constants.js";

/**
 * Retrieves a file from either a remote URL using the Fetch API or from the local file system using the FileSystem API.
 * If the filesystem is available and `env.useCache = true`, the file will be downloaded and cached.
 *
 * @param path_or_repo_id This can be either:
 * - a string, the *model id* of a model repo on huggingface.co.
 * - a path to a *directory* potentially containing the file.
 * @param filename The name of the file to locate in `path_or_repo`.
 * @param fatal Whether to throw an error if the file is not found.
 * @param options An object containing optional parameters.
 * @throws Will throw an error if the file is not found and `fatal` is true.
 * @returns A Promise that resolves with the file content as a buffer.
 */
export async function getModelFile(
	path_or_repo_id: string,
	filename: string,
	fatal: boolean = true,
	options: PretrainedOptions = {}
): Promise<Uint8Array> {
	try {
		if (!env.allowLocalModels && options.local_files_only) {
			throw new Error(
				"Invalid configuration: `env.allowLocalModels` is false, but `local_files_only` is set to true."
			);
		}

		if (!env.allowLocalModels && !env.allowRemoteModels) {
			throw new Error(
				"Both local and remote models are disabled. Enable one of them to proceed."
			);
		}

		// Initiate file retrieval
		dispatchCallback(options.progress_callback, {
			status: "initiate",
			name: path_or_repo_id,
			file: filename,
		});

		// First, check if the a caching backend is available
		// If no caching mechanism available, will download the file every time
		const cache = await initializeCache();

		const fileInfo = constructFilePaths(path_or_repo_id, filename, options);

		// A caching system is available, so we try to get the file from it.
		//  1. We first try to get from cache using the local path. In some environments (like deno),
		//     non-URL cache keys are not allowed. In these cases, `response` will be undefined.
		//  2. If no response is found, we try to get from cache using the remote URL or file system cache.
		const response = await retrieveFileFromCacheOrRemote(
			cache,
			fileInfo,
			fatal
		);

		const buffer = await processFileResponse(
			response,
			path_or_repo_id,
			fileInfo,
			options
		);

		if (cache && fileInfo.cacheKey && response instanceof Response) {
			await addToCache(cache, fileInfo.cacheKey, response, buffer);
		}

		dispatchCallback(options.progress_callback, {
			status: "done",
			name: path_or_repo_id,
			file: filename,
		});

		return buffer;
	} catch (error) {
		console.error("Error getting model:", error);
		if (fatal) throw error;
		return new Uint8Array();
	}
}

/**
 * Creates a new `FileResponse` object.
 * @param filePath
 */
class FileResponse {
	public filePath: string | URL;
	public headers: Headers;
	public exists: boolean;
	public status: number;
	public statusText: string;
	public body: ReadableStream | null = null;

	constructor(filePath: string | URL) {
		this.filePath = filePath;
		this.headers = new Headers();

		this.exists = fs.existsSync(filePath);
		if (this.exists) {
			this.status = 200;
			this.statusText = "OK";

			let stats = fs.statSync(filePath);
			this.headers.set("content-length", stats.size.toString());

			this.updateContentType();

			let self = this;
			this.body = new ReadableStream({
				start(controller) {
					self.arrayBuffer().then((buffer) => {
						controller.enqueue(new Uint8Array(buffer));
						controller.close();
					});
				},
			});
		} else {
			this.status = 404;
			this.statusText = "Not Found";
			this.body = null;
		}
	}

	/**
	 * Updates the 'content-type' header property of the response based on the file extension.
	 */
	public updateContentType(): void {
		try {
			const extension = this.filePath
				.toString()
				.split(".")
				.pop()
				?.toLowerCase();

			// Check if the extension is a valid key in CONTENT_TYPE_MAP
			const contentType =
				extension && extension in CONTENT_TYPE_MAP
					? CONTENT_TYPE_MAP[extension as ContentTypeMap]
					: "application/octet-stream";

			this.headers.set("content-type", contentType);
		} catch (error) {
			console.error("Error updating content type:", error);
		}
	}

	/**
	 * Clone the current FileResponse object.
	 * @returns A new FileResponse object with the same properties as the current object.
	 */
	public clone(): FileResponse {
		let response = new FileResponse(this.filePath);
		response.exists = this.exists;
		response.status = this.status;
		response.statusText = this.statusText;
		response.headers = new Headers(this.headers);
		return response;
	}

	/**
	 * Reads the contents of the file specified by the filePath property and returns a Promise that
	 * resolves with an ArrayBuffer containing the file's contents.
	 * @returns A Promise that resolves with an ArrayBuffer containing the file's contents.
	 * @throws If the file cannot be read.
	 */
	public async arrayBuffer(): Promise<ArrayBuffer> {
		try {
			const data = await fs.promises.readFile(this.filePath);
			return data.buffer;
		} catch (error) {
			console.log("Error getting Array Buffer from File Response", error);
			throw error;
		}
	}

	/**
	 * Reads the contents of the file specified by the filePath property and returns a Promise that
	 * resolves with a Blob containing the file's contents.
	 * @returns A Promise that resolves with a Blob containing the file's contents.
	 * @throws If the file cannot be read.
	 */
	public async blob(): Promise<Blob> {
		try {
			const data = await fs.promises.readFile(this.filePath);
			return new Blob([data], {
				type: this.headers.get("content-type") ?? undefined,
			});
		} catch (error) {
			console.log("Error getting blob from FileResponse", error);
			throw error;
		}
	}

	/**
	 * Reads the contents of the file specified by the filePath property and returns a Promise that
	 * resolves with a string containing the file's contents.
	 * @returns A Promise that resolves with a string containing the file's contents.
	 * @throws If the file cannot be read.
	 */
	public async text(): Promise<string> {
		try {
			return await fs.promises.readFile(this.filePath, "utf8");
		} catch (error) {
			console.log("Error reading text from FileResponse", error);
			throw error;
		}
	}

	/**
	 * Reads the contents of the file specified by the filePath property and returns a Promise that
	 * resolves with a parsed JavaScript object containing the file's contents.
	 *
	 * @returns A Promise that resolves with a parsed JavaScript object containing the file's contents.
	 * @throws If the file cannot be read.
	 */
	public async json(): Promise<object> {
		try {
			return JSON.parse(await this.text());
		} catch (error) {
			console.log("Error reading JSON from FileResponse", error);
			throw error;
		}
	}
}

/**
 * Determines whether the given string is a valid URL.
 * @param input The string or URL to test for validity.
 * @param protocols A list of valid protocols. If specified, the protocol must be in this list.
 * @param validHosts A list of valid hostnames. If specified, the URL's hostname must be in this list.
 * @returns True if the input is a valid URL, false otherwise.
 */
function isValidUrl(
	input: string | URL,
	protocols?: string[],
	validHosts?: string[]
): boolean {
	try {
		const url = typeof input === "string" ? new URL(input) : input;

		const isProtocolValid = !protocols || protocols.includes(url.protocol);
		const isHostValid = !validHosts || validHosts.includes(url.hostname);

		return isProtocolValid && isHostValid;
	} catch {
		return false;
	}
}

/**
 * Helper function to get a file, using either the Fetch API or FileSystem API.
 *
 * @param urlOrPath The URL/path of the file to get.
 * @returns A promise that resolves to a FileResponse object (if the file is retrieved using the FileSystem API), or a Response object (if the file is retrieved using the Fetch API).
 */
export async function getFile(
	urlOrPath: URL | string
): Promise<FileResponse | Response | undefined> {
	try {
		if (env.useFS && !isValidUrl(urlOrPath, ["http:", "https:", "blob:"])) {
			return new FileResponse(urlOrPath);
		} else if (
			typeof process !== "undefined" &&
			process?.release?.name === "node"
		) {
			const IS_CI = !!process.env?.TESTING_REMOTELY;
			const version = env.version;

			const headers = new Headers();
			headers.set("User-Agent", `transformers.js/${version}; is_ci/${IS_CI};`);

			// Check whether we are making a request to the Hugging Face Hub.
			const isHFURL = isValidUrl(
				urlOrPath,
				["http:", "https:"],
				["huggingface.co", "hf.co"]
			);
			if (isHFURL) {
				// If an access token is present in the environment variables,
				// we add it to the request headers.
				// NOTE: We keep `HF_ACCESS_TOKEN` for backwards compatibility (as a fallback).
				const token = process.env?.HF_TOKEN ?? process.env?.HF_ACCESS_TOKEN;
				if (token) {
					headers.set("Authorization", `Bearer ${token}`);
				}
			}
			return fetch(urlOrPath, { headers });
		} else {
			// Running in a browser-environment, so we use default headers
			// NOTE: We do not allow passing authorization headers in the browser,
			// since this would require exposing the token to the client.
			return fetch(urlOrPath);
		}
	} catch (error) {
		console.error("Error in getFile", error);
		return undefined;
	}
}

/**
 * Helper method to handle fatal errors that occur while trying to load a file from the Hugging Face Hub.
 * @param status The HTTP status code of the error.
 * @param remoteURL The URL of the file that could not be loaded.
 * @param fatal Whether to raise an error if the file could not be loaded.
 * @returns Returns `null` if `fatal = true`.
 * @throws If `fatal = false`.
 */
function handleError(status: number, remoteURL: string, fatal: boolean): null {
	if (!fatal) {
		// File was not loaded correctly, but it is optional.
		// TODO in future, cache the response?
		return null;
	}

	const message =
		ERROR_MAPPING[status as ErrorMapping] ??
		`Error (${status}) occurred while trying to load file`;

	throw new Error(`${message}: "${remoteURL}".`);
}

/**
 * Instantiate a `FileCache` object.
 * @param path
 */
class FileCache {
	private path: string;

	constructor(path: string) {
		this.path = path;
	}

	/**
	 * Checks whether the given request is in the cache.
	 */
	public match(request: string): FileResponse | undefined {
		let filePath = path.join(this.path, request);
		let file = new FileResponse(filePath);

		if (file.exists) {
			return file;
		} else {
			return undefined;
		}
	}

	/**
	 * Adds the given response to the cache.
	 */
	public async put(
		request: string,
		response: Response | FileResponse
	): Promise<void> {
		try {
			const buffer = Buffer.from(await response.arrayBuffer());

			let outputPath = path.join(this.path, request);

			await fs.promises.mkdir(path.dirname(outputPath), { recursive: true });
			await fs.promises.writeFile(outputPath, buffer);
		} catch (err) {
			console.warn("An error occurred while writing the file to cache:", err);
		}
	}

	// TODO add the rest?
	// addAll(requests: RequestInfo[]): Promise<void>;
	// delete(request: RequestInfo | URL, options?: CacheQueryOptions): Promise<boolean>;
	// keys(request?: RequestInfo | URL, options?: CacheQueryOptions): Promise<ReadonlyArray<Request>>;
	// match(request: RequestInfo | URL, options?: CacheQueryOptions): Promise<Response | undefined>;
	// matchAll(request?: RequestInfo | URL, options?: CacheQueryOptions): Promise<ReadonlyArray<Response>>;
}

/**
 * Try to get file from cache
 * @param cache The cache to search
 * @param names The names of the item to search for
 * @returns The item from the cache, or undefined if not found.
 */
async function tryCache(
	cache: FileCache | Cache | CustomCache,
	...names: string[]
): Promise<FileResponse | Response | undefined> {
	if (isCustomCache(cache)) {
		for (let name of names) {
			const req = new Request(name);
			let result = cache.match(req);
			if (!result) continue;
			return result;
		}
	} else {
		for (let name of names) {
			let result = cache.match(name);
			if (!result) continue;
			return result;
		}
	}

	// If none found, return undefined
	return undefined;
}

/**
 * Initializes the appropriate cache system based on environment settings.
 */
async function initializeCache(): Promise<
	FileCache | Cache | CustomCache | undefined
> {
	if (env.useBrowserCache) {
		if (typeof caches === "undefined") {
			throw new Error("Browser cache is not available in this environment.");
		}
		try {
			return await caches.open("transformers-cache");
		} catch (e) {
			console.warn("Error opening browser cache:", e);
		}
	} else if (env.useFSCache) {
		return new FileCache(env.cacheDir);
	} else if (env.useCustomCache) {
		if (!env.customCache) {
			throw new Error(
				"`env.useCustomCache` is true, but `env.customCache` is not defined."
			);
		}
		if (!env.customCache.match || !env.customCache.put) {
			throw new Error(
				"`env.customCache` must implement `match` and `put` methods of the Web Cache API."
			);
		}
		return env.customCache as CustomCache;
	}
	return undefined;
}

/**
 * Constructs file paths for local and remote retrieval.
 */
function constructFilePaths(
	path_or_repo_id: string,
	filename: string,
	options: PretrainedOptions
): {
	localPath: string;
	remoteURL: string;
	cacheKey: string;
} {
	const revision = options.revision ?? "main";

	const requestURL = pathJoin(path_or_repo_id, filename);
	const localPath = pathJoin(env.localModelPath, requestURL);

	const remoteURL = pathJoin(
		env.remoteHost,
		env.remotePathTemplate
			.replaceAll("{model}", path_or_repo_id)
			.replaceAll("{revision}", encodeURIComponent(revision)),
		filename
	);

	const cacheKey =
		revision === "main"
			? requestURL
			: pathJoin(path_or_repo_id, revision, filename);

	return { localPath, remoteURL, cacheKey };
}

/**
 * Attempts to retrieve the file from the cache or fetches it remotely.
 */
async function retrieveFileFromCacheOrRemote(
	cache: FileCache | Cache | CustomCache | undefined,
	fileInfo: { localPath: string; remoteURL: string },
	fatal: boolean
): Promise<Response | FileResponse> {
	const { localPath, remoteURL } = fileInfo;

	if (cache) {
		const response = await tryCache(cache, localPath, remoteURL);
		if (response) return response;
	}

	if (env.allowLocalModels) {
		try {
			const file = await getFile(localPath);
			if (!file) throw new Error("No file found");
			return file;
		} catch (e) {
			console.warn(`File not found locally: ${localPath}`, e);
		}
	}

	if (env.allowRemoteModels) {
		const response = await getFile(remoteURL);
		if (!response || response?.status !== 200) {
			handleError(response?.status ?? 404, remoteURL, fatal);
		}
		return response!;
	}

	if (fatal) {
		throw new Error(
			`File not found locally or remotely: ${localPath}, ${remoteURL}`
		);
	}

	return new Response();
}

/**
 * Processes the file response and handles progress reporting.
 */
async function processFileResponse(
	response: Response | FileResponse,
	path_or_repo_id: string,
	fileInfo: { localPath: string },
	options: PretrainedOptions
): Promise<Uint8Array> {
	if (response instanceof FileResponse || !options.progress_callback) {
		return new Uint8Array(await response.arrayBuffer());
	}

	return await readResponse(response, (data) => {
		dispatchCallback(options.progress_callback, {
			status: "progress",
			name: path_or_repo_id,
			file: fileInfo.localPath,
			...data,
		});
	});
}

/**
 * Adds the file to the cache if caching is enabled.
 */
async function addToCache(
	cache: FileCache | Cache | CustomCache,
	cacheKey: string,
	response: Response,
	buffer: Uint8Array
): Promise<void> {
	try {
		if (isCustomCache(cache)) {
			const req = new Request(cacheKey);
			await cache.put(req, new Response(buffer, { headers: response.headers }));
		} else {
			if (response instanceof Response && response.status === 200) {
				await cache.put(
					cacheKey,
					new Response(buffer, { headers: response.headers })
				);
			}
		}
	} catch (err) {
		console.warn(`Unable to add to cache: ${cacheKey}`, err);
	}
}

/**
 * Fetches a JSON file from a given path and file name.
 *
 * @param modelPath The path to the directory containing the file.
 * @param fileName The name of the file to fetch.
 * @param fatal Whether to throw an error if the file is not found.
 * @param options An object containing optional parameters.
 * @returns The JSON data parsed into a JavaScript object.
 * @throws Will throw an error if the file is not found and `fatal` is true.
 */
export async function getModelJSON(
	modelPath: string,
	fileName: string,
	fatal: boolean = true,
	options: PretrainedOptions = {}
): Promise<object> {
	let buffer = await getModelFile(modelPath, fileName, fatal, options);
	if (buffer === null) {
		// Return empty object
		return {};
	}

	let decoder = new TextDecoder("utf-8");
	let jsonData = decoder.decode(buffer);

	return JSON.parse(jsonData);
}

/**
 * Reads and tracks progress when reading a Response object.
 *
 * @param response The Response object to read.
 * @param progressCallback The function to call with progress updates.
 * @returns A Promise that resolves with the Uint8Array buffer.
 */
async function readResponse(
	response: Response | FileResponse,
	progressCallback: (data: {
		progress: number;
		loaded: number;
		total: number;
	}) => void
): Promise<Uint8Array> {
	const contentLength = parseInt(response.headers.get("Content-Length") ?? "0");

	if (!response.body) {
		throw new Error("No body in response");
	}

	if (!contentLength) {
		console.warn(
			"Content-Length not specified; buffer will expand dynamically."
		);
	}

	let total = contentLength || 0;
	let loaded = 0;
	let buffer = new Uint8Array(total);
	const reader = response.body.getReader();

	while (true) {
		const { done, value } = await reader.read();
		if (done) break;

		const newLoaded = loaded + value.length;
		if (newLoaded > total) {
			total = newLoaded;
			const newBuffer = new Uint8Array(total);
			newBuffer.set(buffer); // Copy existing data
			buffer = newBuffer;
		}

		buffer.set(value, loaded);
		loaded = newLoaded;

		progressCallback({
			progress: (loaded / total) * 100,
			loaded,
			total,
		});
	}

	return buffer;
}

/**
 * Joins multiple parts of a path into a single path, handling leading and trailing slashes.
 *
 * @param parts Multiple parts of a path.
 * @returns A string representing the joined path.
 */
function pathJoin(...parts: string[]): string {
	return parts
		.map(
			(part, index) =>
				part
					.replace(/^\/+/, index === 0 ? "" : "/") // Remove leading slashes unless it's the first part
					.replace(/\/+$/, index === parts.length - 1 ? "" : "/") // Remove trailing slashes unless it's the last part
		)
		.join("/");
}

/**
 * Type guard to check if the object is of type CustomCache.
 */
function isCustomCache(cache: unknown): cache is CustomCache {
	return (
		typeof cache === "object" &&
		cache !== null &&
		"match" in cache &&
		"put" in cache &&
		typeof (cache as CustomCache).match === "function" &&
		typeof (cache as CustomCache).put === "function"
	);
}
