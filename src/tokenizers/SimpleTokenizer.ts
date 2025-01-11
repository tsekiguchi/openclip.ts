import path from "path";
import * as zlib from "zlib";
import FS from "fs";
import fs from "fs/promises";
import { decodeHTML } from "entities";
import * as ort from "onnxruntime-web";
import natural from "natural";

const defaultBPEPath = path.join(
	import.meta.dirname,
	"bpe_simple_vocab_16e6.txt.gz"
);

interface SimpleTokenizerConfig {
	bpePath: string;
	additionalSpecialTokens?: string[];
	contextLength: number;
	clean: CleanFunctions;
	reductionMask: ReductionFunctions;
}

const defaultSTConfig: SimpleTokenizerConfig = {
	bpePath: defaultBPEPath,
	contextLength: 77, // default context length for OpenAI CLIP
	clean: "lower",
	reductionMask: null,
};

type CleanFunction = (text: string) => string;
type ReductionFunction = (text: string | string[]) => ort.Tensor;
type CleanFunctions = "canonicalize" | "lower" | "whitespace";
type ReductionFunctions = "simple" | "random" | "shuffle" | "syntax" | null;

type Tuple = [string, string];
type Vocab = string[];
type ByteEncoder = Map<number, string>;
type ByteDecoder = Map<string, number>;
type Encoder = Map<string, number>;
type Decoder = Map<number, string>;

/**
 * Returns list of utf-8 byte and a corresponding list of unicode strings.
 * The reversible bpe codes work on unicode strings.
 * This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
 * When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
 * This is a significant percentage of your normal, say, 32K bpe vocab.
 * To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
 * And avoids mapping to whitespace/control characters the bpe code barfs on.
 */
export class SimpleTokenizer {
	private config: SimpleTokenizerConfig;
	private sotToken = "<start_of_text>";
	private eotToken = "<end_of_text>";
	private byteEncoder: ByteEncoder;
	private byteDecoder: ByteDecoder;
	private encoder: Encoder;
	private decoder: Decoder;
	private bpeRanks: Map<string, number>;
	private cache: Map<string, string>;
	private patterns: RegExp;
	private vocabSize: number;
	private allSpecialIds: number[];
	private sotTokenId: number;
	private eotTokenId: number;
	private contextLength: number;
	private cleanFn: CleanFunction;
	private reductionFn: ReductionFunction | null;

	constructor(config: Partial<SimpleTokenizerConfig> = {}) {
		// Merge provided config with default config
		this.config = {
			...defaultSTConfig,
			...config,
		};

		this.contextLength = this.config.contextLength;

		// Create encoder/decoder for bytes
		const { byteEncoder, byteDecoder } = this.createByteEncoderDecoder();
		this.byteEncoder = byteEncoder;
		this.byteDecoder = byteDecoder;

		const merges = this.readBPEVocab(this.config.bpePath);
		this.bpeRanks = new Map<string, number>(
			merges.map((merge, index) => [merge, index])
		);

		const vocab = this.createVocab(merges);

		// Add special tokens, plus any additional from config
		const specialTokens = this.getSpecialTokens();
		vocab.push(...specialTokens);
		this.cache = new Map<string, string>(
			specialTokens.map((token) => [token, token])
		);

		const special = specialTokens.join("|");

		// PYTHON:
		// self.pat = re.compile(
		// 		special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
		// 		re.IGNORECASE,
		// )
		this.patterns = new RegExp(
			special +
				"|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+",
			"ugi" // u for unicode, g for global, i for case insensitive
		);

		// Create encoder / decoder for vocab
		const { encoder, decoder } = this.createEncoderDecoder(vocab);
		this.encoder = encoder;
		this.decoder = decoder;
		this.vocabSize = encoder.keys.length;

		this.allSpecialIds = specialTokens
			.map((t) => encoder.get(t))
			.filter((t) => t !== undefined);
		this.sotTokenId = encoder.get(this.sotToken)!;
		this.eotTokenId = encoder.get(this.eotToken)!;

		this.cleanFn = this.getCleanFunction(this.config.clean);
		this.reductionFn = this.config.reductionMask
			? this.getReductionMaskFn(this.config.reductionMask)
			: null;
	}

	/**
	 * Returns the tokenized representation of given input string(s)
	 *
	 * Python version has context length as an option -
	 * however that has already been set in initialization.
	 * @param texts an input string or list of strings to tokenize
	 * @returns An ONNX Runtime Tensor (ort.Tensor) in the shape of [number of input stings, contextLength]
	 */
	public tokenize(texts: string | string[]): ort.Tensor {
		if (typeof texts === "string") {
			texts = [texts];
		}

		// Default tokenization with truncation
		const allTokens = texts.map((text) => [
			this.sotTokenId,
			...this.encode(text),
			this.eotTokenId,
		]);

		// Create result array with zeros
		const result: number[][] = Array.from({ length: allTokens.length }, () =>
			new Array(this.contextLength).fill(0)
		);

		const flatResult = new Int32Array(texts.length * this.contextLength);

		for (let i = 0; i < allTokens.length; i++) {
			let tokens = allTokens[i];
			if (tokens.length > this.contextLength) {
				tokens = tokens.slice(0, this.contextLength); // Truncate
				tokens[tokens.length - 1] = this.eotTokenId; // Set the last token to EOT
			}

			for (let j = 0; j < tokens.length; j++) {
				flatResult[i * this.contextLength + j] = tokens[j];
			}
		}

		return new ort.Tensor("int32", flatResult, [
			texts.length,
			this.contextLength,
		]);
	}

	private createByteEncoderDecoder(): {
		byteEncoder: ByteEncoder;
		byteDecoder: ByteDecoder;
	} {
		// Create initial arrays of byte values
		const bytes: number[] = [
			// ASCII printable characters (33-126)
			...Array.from(
				{ length: this.ord("~") - this.ord("!") + 1 },
				(_, i) => this.ord("!") + i
			),

			// Latin-1 characters (161-172)
			...Array.from(
				{ length: this.ord("¬") - this.ord("¡") + 1 },
				(_, i) => this.ord("¡") + i
			),
			// Latin-1 characters (174-255)
			...Array.from(
				{ length: this.ord("ÿ") - this.ord("®") + 1 },
				(_, i) => this.ord("®") + i
			),
		];

		// Make a copy of the initial array
		const copyBytes: number[] = [...bytes];

		// Fill in the gaps with values starting from 256
		let n = 0;
		for (let b = 0; b < 256; b++) {
			if (!bytes.includes(b)) {
				bytes.push(b);
				copyBytes.push(256 + n);
				n += 1;
			}
		}

		// Convert numbers to Unicode characters and create the mapping
		const byteEncoder = new Map<number, string>();
		for (let i = 0; i < bytes.length; i++) {
			byteEncoder.set(bytes[i], String.fromCharCode(copyBytes[i]));
		}

		const byteDecoder = new Map<string, number>();
		for (let i = 0; i < bytes.length; i++) {
			byteDecoder.set(String.fromCharCode(copyBytes[i]), bytes[i]);
		}

		return { byteEncoder, byteDecoder };
	}

	// Helper function to get the Unicode code point of a character
	private ord(char: string): number {
		return char.charCodeAt(0);
	}

	/**
	 * Create an array of tuples from the merges file
	 * Note: Tuples don't work for lookup in maps, so we keep merges as strings with spaces separating
	 * @param bpePath Path to the bpe simple vocab file
	 * @returns Merges string[]
	 */
	private readBPEVocab(bpePath: string): string[] {
		const buffer = FS.readFileSync(bpePath);
		const content = zlib.gunzipSync(buffer).toString("utf-8");
		const lines = content.split("\n");

		// Slice the array (equivalent to merges[1:49152-256-2+1])
		// The first line of the merges.txt is the # version, so we leave that out
		const slicedLines = lines.slice(1, 49152 - 256 - 2 + 1);

		// Convert each line into a tuple of strings
		const merges: string[] = slicedLines.filter((line) => line.trim() !== ""); // Skip empty lines
		// .map((line) => {
		// 	const [first, second] = line.split(/\s+/);
		// 	return [first, second] as [string, string];
		// });

		return merges;
	}

	private createVocab(merges: string[]): Vocab {
		// Get the unicode characters created earlier
		// Python:
		// vocab = list(bytes_to_unicode().values())
		
		let vocab: string[] = []
		vocab.push(...this.byteEncoder.values())

		
		// Create a vocab list, once with the characters, and again with </w>
		// Python:
		// vocab = vocab + [v+'</w>' for v in vocab]
		vocab = [...vocab, ...vocab.map((v) => v + "</w>")];
		
		
		for (const merge of merges) {
			// takes all merges, merges together, adds to vocab
			const noSpaces = merge.replace(/ /g, ""); // Removes all spaces
			vocab.push(noSpaces);
		}

		const index = vocab.findIndex((word) => word == 'tonight</w>')

		return vocab;
	}

	private getSpecialTokens(): string[] {
		// Add special tokens, plus any additional from config
		const specialTokens = [this.sotToken, this.eotToken];
		if (this.config.additionalSpecialTokens)
			specialTokens.push(...this.config.additionalSpecialTokens);
		return specialTokens;
	}

	private createEncoderDecoder(vocab: Vocab): {
		encoder: Encoder;
		decoder: Decoder;
	} {
		// Map to encoder / decoder
		const encoder = new Map<string, number>(
			vocab.map((token, index) => [token, index])
		);
		const decoder = new Map<number, string>(
			vocab.map((token, index) => [index, token])
		);

		return { encoder, decoder };
	}

	private getCleanFunction(cleanFunction: CleanFunctions): CleanFunction {
		if (cleanFunction == "canonicalize") {
			return this.cleanCanonicalize;
		} else if (cleanFunction == "lower") {
			return this.cleanLower;
		} else if (cleanFunction == "whitespace") {
			return this.cleanWhitespace;
		} else {
			throw new Error("Invalid cleaning function	");
		}
	}

	// Before: "Here&#39;s some text with &quot;quotes&quot; &amp; special characters"
	// After:  "Here's some text with \"quotes\" & special characters"
	private basicClean: CleanFunction = (text: string) => {
		text = text.normalize("NFKC");
		text = decodeHTML(decodeHTML(text));
		return text.trim();
	};

	// Before: "hello      world   from		TypeScript"
	// After:  "hello world from TypeScript"
	private whitespaceClean(text: string): string {
		text = text.split(/\s+/).join(" ");
		return text.trim();
	}

	private cleanCanonicalize: CleanFunction = (text: string) => {
		return this.canonicalizeText(this.basicClean(text));
	};

	private cleanLower: CleanFunction = (text: string) => {
		return this.whitespaceClean(this.basicClean(text)).toLowerCase();
	};

	private cleanWhitespace: CleanFunction = (text: string) => {
		return this.whitespaceClean(this.basicClean(text));
	};

	/**
	 * Returns canonicalized `text` (lowercase and punctuation removed).
	 * Before: "Hello, world!"
	 * After: "hello world"
	 * @param text string to be canonicalized
	 * @param options.keepPunctuationExactString If provided, this exact string is kept.
	 *     For example, providing '{}' will keep any occurrences of '{}'
	 *     but will still remove '{' and '}' that appear separately.
	 * @param options.transPunctuation A map of characters to remove.
	 * @returns
	 */
	private canonicalizeText(
		text: string,
		options: {
			keepPunctuationExactString?: string;
			transPunctuation?: Record<string, string>;
		} = {}
	): string {
		const {
			keepPunctuationExactString = undefined,
			transPunctuation = Object.fromEntries(
				[..."!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"].map((char) => [char, ""])
			),
		} = options;

		// Replace underscores with spaces
		text = text.replace(/_/g, " ");

		if (keepPunctuationExactString) {
			text = text
				.split(keepPunctuationExactString)
				.map((part) =>
					part.replace(
						new RegExp(`[${Object.keys(transPunctuation).join("")}]`, "g"),
						""
					)
				)
				.join(keepPunctuationExactString);
		} else {
			text = text.replace(
				new RegExp(`[${Object.keys(transPunctuation).join("")}]`, "g"),
				""
			);
		}

		// Convert to lowercase
		text = text.toLowerCase();

		// Normalize spaces
		text = text.replace(/\s+/g, " ").trim();

		return text;
	}

	/**
	 * Byte Pair Encoding
	 * @param token
	 * @returns
	 */
	public bpe(token: string): string {
		const cacheToken = this.cache.get(token);
		if (cacheToken) return cacheToken;

		// console.log('incoming token', token)
	
		let word = [...token.slice(0, -1), token.slice(-1) + "</w>"];
		let pairs = this.getPairs(word);

		// console.log('pairs', pairs)
	
		if (pairs.length === 0) {
			return token + "</w>";
		}
	
		while (true) {
			const bigram = pairs.reduce((best, pair) => {
				const rank = this.bpeRanks.get(pair) ?? Infinity;
				const bestRank = this.bpeRanks.get(best) ?? Infinity;
				return rank < bestRank ? pair : best;
			});
	
			
	
			if (!this.bpeRanks.has(bigram)) {
				break;
			}
	
			const [first, second] = bigram.split(" ")
			const newWord: string[] = [];
			let i = 0;
	
			while (i < word.length) {
				const j = word.indexOf(first, i);
	
				if (j === -1) {
					newWord.push(...word.slice(i));
					break;
				}
	
				newWord.push(...word.slice(i, j));
	
				if (word[j] === first && j < word.length - 1 && word[j + 1] === second) {
					newWord.push(first + second);
					i = j + 2;
				} else {
					newWord.push(word[j]);
					i = j + 1;
				}
			}
	
			word = newWord

			if (word.length === 1) {
				break;
			}
	
			pairs = this.getPairs(word);
		}
	
		const result = word.join(" ");
		this.cache.set(token, result);
		return result;
	}
	

	/**
	 * Transform text into token ids
	 * @param text
	 * @returns number[] an array of token ids
	 */
	public encode(text: string): number[] {
		// const textArray = typeof text === "string" ? [texts] : texts;

		const bpeTokens: number[] = [];
		text = this.cleanFn(text);

		const tokens = text.match(this.patterns) || [];

		for (let i = 0; i < tokens.length; i++) {
			const token = tokens[i];
			const tokensArray: string[] = [];

			const encodedToken = Array.from(token, (char) => {
				const byte = char.charCodeAt(0);
				const encoded = this.byteEncoder.get(byte);
				if (!encoded) throw new Error(`Byte ${byte} not found in byteEncoder`);
				return encoded;
			}).join("");

			const bpeParts = this.bpe(encodedToken).split(" ");

			for (const bpeToken of bpeParts) {

				console.log('bpetoken', bpeToken)
				const id = this.encoder.get(bpeToken);

				console.log("id", id);
				if (id) bpeTokens.push(id);
			}
		}

		return bpeTokens;
	}

	public decode(tokens: number[]) {
		// Decode tokens into string
		// Filter for falsy values
		const textArray: string[] = [];
		for (const token of tokens) {
			const t = this.decoder.get(token);
			if (t) textArray.push(t);
		}
		const text = textArray.join("");

		// Get bytes from decoder
		// Filter for falsy values
		const byteArray: number[] = [];
		for (const char of text) {
			const b = this.byteDecoder.get(char);
			if (b) byteArray.push(b);
		}

		return new TextDecoder("utf-8", { fatal: false })
			.decode(new Uint8Array(byteArray))
			.replace("</w>", " ");
	}

	/**
	 * Return a set of symbol pairs in a word.
	 * The word is represented as an array of symbols (variable-length strings).
	 * @param word
	 * @returns [string, string]
	 */
	private getPairs(word: string[]): string[] {
		const pairs: string[] = [];

		if (word.length < 2) {
			return []; // No pairs if the word has less than two symbols
		}

		let prevChar = word[0];
		for (let i = 1; i < word.length; i++) {
			const char = word[i];
			pairs.push(`${prevChar} ${char}`);
			prevChar = char;
		}

		return pairs;
	}

	private getReductionMaskFn(
		reductionType: ReductionFunctions
	): ReductionFunction {
		if (reductionType == "simple") {
			return this.simpleMaskTokenize;
		} else if (reductionType == "random") {
			return (texts: string | string[]) => this.randomMaskTokenize(texts);
		} else if (reductionType == "shuffle") {
			return (texts: string | string[]) => this.randomMaskTokenize(texts, true);
		} else if (reductionType == "syntax") {
			return this.syntaxMaskTokenize;
		} else {
			throw new Error("Reduction Type Function not found");
		}
	}

	private simpleMaskTokenize: ReductionFunction = (
		texts: string | string[]
	): ort.Tensor => {
		// Convert single string to array if necessary
		const textArray = typeof texts === "string" ? [texts] : texts;

		// Encode all texts
		const allTokens = textArray.map((text) => this.encode(text));

		const resultData = new Int32Array(
			textArray.length * this.contextLength
		).fill(0);

		// Tokenize and pad/truncate each sequence
		for (let i = 0; i < allTokens.length; i++) {
			let tokens = allTokens[i];
			const numTokens = tokens.length;

			// Truncate if necessary
			if (numTokens > this.contextLength - 2) {
				// 2 for sot and eot tokens
				const numKeep = this.contextLength - 2;
				const startIndex = Math.floor(
					Math.random() * (numTokens - numKeep + 1)
				); // inclusive range
				tokens = tokens.slice(startIndex, startIndex + numKeep);
			}

			// Add special tokens
			tokens = [this.sotTokenId, ...tokens, this.eotTokenId];

			// Copy tokens into the result tensor
			for (let j = 0; j < tokens.length; j++) {
				resultData[i * this.contextLength + j] = tokens[j];
			}
		}

		return new ort.Tensor("int32", resultData, [
			textArray.length,
			this.contextLength,
		]);
	};

	/**
	 * Tokenizes and masks text sequences with optional random shuffling.
	 *
	 * @param texts - A single string or a list of strings to be tokenized.
	 * @param contextLength - The maximum context length (sequence length).
	 * @param sotTokenId - Start-of-text token ID.
	 * @param eotTokenId - End-of-text token ID.
	 * @param encodeFn - A function that encodes a string into a sequence of token IDs.
	 * @param shuffle - Whether to shuffle the tokens randomly.
	 * @returns A tensor with tokenized and masked sequences.
	 */
	private randomMaskTokenize = (
		texts: string | string[],
		shuffle: boolean = false
	): ort.Tensor => {
		// Ensure texts is an array
		const textArray = typeof texts === "string" ? [texts] : texts;

		// Encode all texts
		const allTokens = textArray.map((text) => this.encode(text));

		// Create a tensor filled with zeros for the result
		const resultData = new Int32Array(
			textArray.length * this.contextLength
		).fill(0);

		for (let i = 0; i < allTokens.length; i++) {
			let tokens = allTokens[i];
			let numTokens = tokens.length;

			if (numTokens > this.contextLength - 2) {
				// 2 for sot and eot tokens
				const numKeep = this.contextLength - 2;

				// Generate random indices for masking
				let indices = Array.from({ length: numTokens }, (_, idx) => idx).sort(
					() => Math.random() - 0.5
				); // Shuffle

				indices = indices.slice(0, numKeep);

				if (!shuffle) {
					indices.sort((a, b) => a - b); // Sort indices if not shuffled
				}

				tokens = indices.map((index) => tokens[index]);
				numTokens = numKeep;
			}

			// Add special tokens
			const startIdx = i * this.contextLength;
			resultData[startIdx] = this.sotTokenId; // Start-of-text token
			tokens.forEach((token, idx) => {
				resultData[startIdx + 1 + idx] = token; // Main tokens
			});
			resultData[startIdx + 1 + numTokens] = this.eotTokenId; // End-of-text token
		}

		// Create and return the tensor
		return new ort.Tensor("int32", resultData, [
			textArray.length,
			this.contextLength,
		]);
	};

	/**
	 * Tokenizes and syntax masks text sequences before processing.
	 *
	 * @param texts - A single string or a list of strings to be tokenized.
	 * @param contextLength - The maximum context length (sequence length).
	 * @param sotTokenId - Start-of-text token ID.
	 * @param eotTokenId - End-of-text token ID.
	 * @param encodeFn - A function that encodes a string into a sequence of token IDs.
	 * @returns A tensor with tokenized and syntax-masked sequences.
	 */
	private syntaxMaskTokenize: ReductionFunction = (
		texts: string | string[]
	): ort.Tensor => {
		// Ensure texts is an array
		const textArray = typeof texts === "string" ? [texts] : texts;

		// Function to determine syntax order
		const getOrder = (tag: string): number => {
			if (tag.startsWith("NN")) return 1; // Nouns
			if (tag.startsWith("JJ")) return 2; // Adjectives
			if (tag.startsWith("VB")) return 3; // Verbs
			return 4; // Other parts of speech
		};

		// Syntax masking
		const newTexts = textArray.map((text) => {
			const tokenizer = new natural.WordTokenizer();
			const tokens = tokenizer.tokenize(text);

			const language = "EN";
			const defaultCategory = "N";
			const defaultCategoryCapitalized = "NNP";

			const lexicon = new natural.Lexicon(
				language,
				defaultCategory,
				defaultCategoryCapitalized
			);
			const ruleSet = new natural.RuleSet("EN");
			const tagger = new natural.BrillPOSTagger(lexicon, ruleSet);

			const posTags = tagger.tag(tokens);

			const orderList = posTags.taggedWords.map(({ tag }) => getOrder(tag));

			const sortedIndices = orderList
				.map((order, index) => ({ order, index }))
				.sort((a, b) => a.order - b.order)
				.map((item) => item.index);

			const sampledIndices = sortedIndices
				.slice(0, this.contextLength - 2)
				.sort((a, b) => a - b);
			const sampledTokens = sampledIndices.map((index) => tokens[index]);

			return sampledTokens.join(" ");
		});

		// Encode all texts
		const allTokens = newTexts.map((text) => [
			this.sotTokenId,
			...this.encode(text),
			this.eotTokenId,
		]);

		// Create a tensor filled with zeros for the result
		const resultData = new Int32Array(
			textArray.length * this.contextLength
		).fill(0);

		allTokens.forEach((tokens, i) => {
			if (tokens.length > this.contextLength) {
				tokens = tokens.slice(0, this.contextLength);
				tokens[this.contextLength - 1] = this.eotTokenId; // Ensure the last token is EOT
			}

			const startIdx = i * this.contextLength;
			tokens.forEach((token, idx) => {
				resultData[startIdx + idx] = token;
			});
		});

		return new ort.Tensor("int32", resultData, [
			textArray.length,
			this.contextLength,
		]);
	};
}
