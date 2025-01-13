export const HF_FILES = {
	WEIGHTS_NAME: "open_clip_pytorch_model.bin",
	SAFE_WEIGHTS_NAME: "open_clip_model.safetensors",
	CONFIG_NAME: "open_clip_config.json",
	ONNX_FP32_NAME: "model.onnx",
} as const;

export const ERROR_MAPPING = Object.freeze({
	// 4xx errors (https://developer.mozilla.org/en-US/docs/Web/HTTP/Status#client_error_responses)
	400: "Bad request error occurred while trying to load file",
	401: "Unauthorized access to file",
	403: "Forbidden access to file",
	404: "Could not locate file",
	408: "Request timeout error occurred while trying to load file",

	// 5xx errors (https://developer.mozilla.org/en-US/docs/Web/HTTP/Status#server_error_responses)
	500: "Internal server error error occurred while trying to load file",
	502: "Bad gateway error occurred while trying to load file",
	503: "Service unavailable error occurred while trying to load file",
	504: "Gateway timeout error occurred while trying to load file",
} as const);

export type ErrorMapping = keyof typeof ERROR_MAPPING;

/**
 * Mapping from file extensions to MIME types.
 */
export const CONTENT_TYPE_MAP = Object.freeze({
	txt: "text/plain",
	html: "text/html",
	css: "text/css",
	js: "text/javascript",
	json: "application/json",
	png: "image/png",
	jpg: "image/jpeg",
	jpeg: "image/jpeg",
	gif: "image/gif",
} as const);

export type ContentTypeMap = keyof typeof CONTENT_TYPE_MAP;
