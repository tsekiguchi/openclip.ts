import { ModelConfig } from "../model";

// Wrapper class for AutoTokenizer from HF Transformers
export class HFTokenizer {
	private tokenizerName: string
	private readonly contextLength: number
	private clean = 'whitespace'
	private stripSepToken = false
	private language: string | null = null
	private cacheDir: string | null = null
	private keywordArgs: string | null = null

	constructor(config: ModelConfig) {
		// performed check for name before init
		this.tokenizerName = 	config.text_cfg.hf_tokenizer_name!
		this.contextLength = config.text_cfg.context_length
	}


}