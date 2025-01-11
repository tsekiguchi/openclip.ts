import path from "path";
import { MobileCLIPBConfig, MobileCLIPConfig } from "./model_types/MobileCLIP";
import { SimpleTokenizer } from "./tokenizers/SimpleTokenizer";
import { HFTokenizer } from "./tokenizers/HFTokenizer";

interface RuntimeConfig {
	device: "cpu" | "gpu";
	dtype: "fp32" | "fp16" | "q8";
	forceQuickGelu?: boolean;
	forceCustomText?: boolean;
}

export type ModelConfig = MobileCLIPConfig;

type Tokenizer = SimpleTokenizer | HFTokenizer

type ModelName = "openAI" | "mobileCLIP" | string

const HF_HUB_PREFIX = "hf-hub:";
const MODEL_CONFIG_DIR = path.join(import.meta.filename, "model_configs");

export class Model {
	private modelName: string;
	private device: string;
	private dtype: string;
	private quickGelu = false;
	private forceCustomText = false;
	private contextLength: number | null = null;

	constructor(modelName: ModelName, deviceConfig: RuntimeConfig) {
		const config = this.getModelConfigFromJSON(modelName);

		this.modelName = modelName;
		this.device = deviceConfig.device;
		this.dtype = deviceConfig.dtype;
		this.quickGelu = deviceConfig.forceQuickGelu ?? false;
		this.forceCustomText = deviceConfig.forceCustomText ?? false;
	}

	private getModelConfigFromJSON(modelName: ModelName): ModelConfig {
		// If getting model from hf hub, then get config from there
		// Want 'open_clip_config.json'
		// Else get the config from a local directory / path
		// Throw error if config for model not found

		const config = MobileCLIPBConfig

		if (!this.contextLength) {
			this.contextLength = config.text_cfg.context_length
		}

		return config;
	}

	private getTokenizer(config: ModelConfig): Tokenizer {
		if (config.text_cfg.hf_tokenizer_name) {
			return new HFTokenizer(config)
		}

		return new SimpleTokenizer()
	}
}

const model = new Model("mobileCLIP", { device: "cpu", dtype: "fp32" });

// Sample open_clip_config.json file
// {
// 	"model_cfg": {
// 			"embed_dim": 512,
// 			"vision_cfg": {
// 					"image_size": 224,
// 					"layers": 12,
// 					"width": 768,
// 					"patch_size": 32
// 			},
// 			"text_cfg": {
// 					"context_length": 77,
// 					"vocab_size": 49408,
// 					"width": 512,
// 					"heads": 8,
// 					"layers": 12
// 			}
// 	},
// 	"preprocess_cfg": {
// 			"mean": [
// 					0.48145466,
// 					0.4578275,
// 					0.40821073
// 			],
// 			"std": [
// 					0.26862954,
// 					0.26130258,
// 					0.27577711
// 			]
// 	}
// }
