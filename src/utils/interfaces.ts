import { PretrainedConfig } from "../configs.js";
import { DeviceType } from "./devices.js";
import { DataType } from "./dtypes.js";
import { ProgressCallback } from "./core.js";
import { InferenceSession } from "onnxruntime-web";
import { SimpleTokenizer } from "../tokenizers/SimpleTokenizer.js";
import { HFTokenizer } from "../tokenizers/HFTokenizer.js";
import { VisionConfig, TextConfig } from "../model_types/MobileCLIP.js";
import * as ort from 'onnxruntime-web'

export interface PretrainedOptions {
	progress_callback?: ProgressCallback;
	config?: PretrainedConfig;
	cache_dir?: string;
	local_files_only?: boolean;
	revision?: string;
}

export interface ModelSpecificPretrainedOptions {
	subfolder?: string;
	model_file_name?: string;
	device?: DeviceType | Record<string, DeviceType>;
	dtype?: DataType | Record<string, DataType>;
	use_external_data_format?: boolean | Record<string, boolean>;
	session_options?: InferenceSession.SessionOptions;
}

export type PretrainedModelOptions = PretrainedOptions &
	ModelSpecificPretrainedOptions;

export interface CustomCache {
	match: (request: Request) => Promise<Response | undefined>;
	put: (request: Request, response: Response) => Promise<void>;
}

export interface ModelConfig {
	embed_dim: number;
	quick_gelu?: boolean;
	vision_cfg: VisionConfig;
	text_cfg: TextConfig;
	custom_text?: boolean;
}

export interface OpenCLIPConfig {
	model_cfg: ModelConfig;
	preprocess_cfg: {
		mean: number[];
		std: number[];
	};
}

export type ModelType =
	| "llava"
	| "paligemma"
	| "florence2"
	| "llava_onevision"
	| "idefics3"
	| "moondream1"
	| "musicgen"
	| "multi_modality"
	// Decoder-only models
	| "gpt2"
	| "gptj"
	| "jais"
	| "codegen"
	| "gpt_bigcode"
	| "gpt_neox"
	| "stablelm"
	| "opt"
	| "falcon"
	| "llama"
	| "olmo"
	| "olmo2"
	| "mobilellm"
	| "granite"
	| "cohere"
	| "mistral"
	| "starcoder2"
	| "qwen2"
	| "qwen2_vl"
	| "phi"
	| "phi3"
	| "phi3_v"
	| "gemma"
	| "gemma2"
	| "openelm"
	| "gpt_neo"
	| "donut-swin"
	| "bloom"
	| "mpt"
	| "exaone"
	// Encoder-decoder models
	| "t5"
	| "mt5"
	| "longt5"
	| "bart"
	| "mbart"
	| "marian"
	| "whisper"
	| "m2m_100"
	| "blenderbot"
	| "blenderbot-small"
	| "florence2_language"
	| "speecht5"
	| "trocr"
	| "musicgen_decoder"
	| "moonshine"
	| "vision-encoder-decoder";

export interface PretrainedConfigJSON {
	model_type: ModelType;
	text_config?: PretrainedConfigJSON;
	phi_config?: PretrainedConfigJSON;
	encoder?: PretrainedConfigJSON;
	decoder?: PretrainedConfigJSON;
	language_config?: PretrainedConfigJSON;
	num_heads?: number;
	num_layers?: number;
	hidden_size?: number;
	num_attention_heads?: number;
	dim_kv?: number;
	multi_query?: boolean;

	is_encoder_decoder?: boolean;

	encoder_hidden_size?: number;
	num_encoder_layers?: number;
	num_encoder_heads?: number;
	encoder_dim_kv?: number;

	decoder_hidden_size?: number;
	num_decoder_layers?: number;
	num_decoder_heads?: number;
	decoder_dim_kv?: number;
}

export interface NormalizedConfig {
	model_type: ModelType;
	is_encoder_decoder?: boolean;
	[key: string]: unknown; // Allows for normalized keys
}

export interface ModelInitOptions {
	device?: "cpu" | "gpu";
	dtype?: "fp32" | "fp16" | "q8";
	forceQuickGelu?: boolean;
	forceCustomText?: boolean;
	forcePatchDropout?: number;
  forceImageSize?: number;
  forcePreprocessCfg?: Record<string, any>;
  pretrainedImage?: boolean;
  pretrainedHf?: boolean;
  cacheDir?: string;
  outputDict?: boolean;
  requirePretrained?: boolean;
  loadWeightsOnly?: boolean;
  // modelKwargs?: Record<string, any>;
}

type RGBTuple = readonly [number, number, number];

export interface PreProcessorConfig {
	mean: RGBTuple
	std: RGBTuple
	size?: number | number[];
	mode?: "RGB" | "RGBA"; // add other modes if there are (black and white, alpha)
	interpolation?: "bicubic" | "bilinear";
	resize_mode?: "squash" | "shortest",
	fill_color?: number
}

export interface PreProcessorConfigs {
	openAI: PreProcessorConfig;
	imageNet: PreProcessorConfig;
	inception: PreProcessorConfig;
	mobileCLIP: PreProcessorConfig;
}

// export type ModelConfig = MobileCLIPConfig;

export type Tokenizer = SimpleTokenizer | HFTokenizer;

export interface TextInput {
	[k: string]: ort.Tensor
}

export interface TextEmbeddings {
	[k: string]: ort.Tensor
}

export interface ImageInput {
	[k: string]: ort.Tensor
}

export interface ImageEmbeddings {
	[k: string]: ort.Tensor
}

// export type ModelName = "openAI" | "mobileCLIP"

export type ModelName =
	| "coca_base"
	| "coca_roberta-ViT-B-32"
	| "coca_ViT-B-32"
	| "coca_ViT-L-14"
	| "convnext_base_w_320"
	| "convnext_base_w"
	| "convnext_base"
	| "convnext_large_d_320"
	| "convnext_large_d"
	| "convnext_large"
	| "convnext_small"
	| "convnext_tiny"
	| "convnext_xlarge"
	| "convnext_xxlarge_320"
	| "convnext_xxlarge"
	| "EVA01-g-14-plus"
	| "EVA01-g-14"
	| "EVA02-B-16"
	| "EVA02-E-14-plus"
	| "EVA02-E-14"
	| "EVA02-L-14-336"
	| "EVA02-L-14"
	| "MobileCLIP-B"
	| "MobileCLIP-S1"
	| "MobileCLIP-S2"
	| "mt5-base-ViT-B-32"
	| "mt5-xl-ViT-H-14"
	| "nllb-clip-base-siglip"
	| "nllb-clip-base"
	| "nllb-clip-large-siglip"
	| "nllb-clip-large"
	| "RN50-quickgelu"
	| "RN50"
	| "RN50x4-quickgelu"
	| "RN50x4"
	| "RN50x16-quickgelu"
	| "RN50x16"
	| "RN50x64-quickgelu"
	| "RN50x64"
	| "RN101-quickgelu"
	| "RN101"
	| "roberta-ViT-B-32"
	| "swin_base_patch4_window7_224"
	| "vit_medium_patch16_gap_256"
	| "vit_relpos_medium_patch16_cls_224"
	| "ViT-B-16-plus-240"
	| "ViT-B-16-plus"
	| "ViT-B-16-quickgelu"
	| "ViT-B-16-SigLIP-256"
	| "ViT-B-16-SigLIP-384"
	| "ViT-B-16-SigLIP-512"
	| "ViT-B-16-SigLIP-i18n-256"
	| "ViT-B-16-SigLIP"
	| "ViT-B-16"
	| "ViT-B-32-256"
	| "ViT-B-32-plus-256"
	| "ViT-B-32-quickgelu"
	| "ViT-B-32"
	| "ViT-bigG-14-CLIPA-336"
	| "ViT-bigG-14-CLIPA"
	| "ViT-bigG-14-quickgelu"
	| "ViT-bigG-14"
	| "ViT-e-14"
	| "ViT-g-14"
	| "ViT-H-14-378-quickgelu"
	| "ViT-H-14-378"
	| "ViT-H-14-CLIPA-336"
	| "ViT-H-14-CLIPA"
	| "ViT-H-14-quickgelu"
	| "ViT-H-14"
	| "ViT-H-16"
	| "ViT-L-14-280"
	| "ViT-L-14-336-quickgelu"
	| "ViT-L-14-336"
	| "ViT-L-14-CLIPA-336"
	| "ViT-L-14-CLIPA"
	| "ViT-L-14-quickgelu"
	| "ViT-L-14"
	| "ViT-L-16-320"
	| "ViT-L-16-SigLIP-256"
	| "ViT-L-16-SigLIP-384"
	| "ViT-L-16"
	| "ViT-M-16-alt"
	| "ViT-M-16"
	| "ViT-M-32-alt"
	| "ViT-M-32"
	| "ViT-S-16-alt"
	| "ViT-S-16"
	| "ViT-S-32-alt"
	| "ViT-S-32"
	| "ViT-SO400M-14-SigLIP-378"
	| "ViT-SO400M-14-SigLIP-384"
	| "ViT-SO400M-14-SigLIP"
	| "ViT-SO400M-16-SigLIP-i18n-256"
	| "ViTamin-B-LTT"
	| "ViTamin-B"
	| "ViTamin-L-256"
	| "ViTamin-L-336"
	| "ViTamin-L-384"
	| "ViTamin-L"
	| "ViTamin-L2-256"
	| "ViTamin-L2-336"
	| "ViTamin-L2-384"
	| "ViTamin-L2"
	| "ViTamin-S-LTT"
	| "ViTamin-S"
	| "ViTamin-XL-256"
	| "ViTamin-XL-336"
	| "ViTamin-XL-384"
	| "xlm-roberta-base-ViT-B-32"
	| "xlm-roberta-large-ViT-H-14"
	| "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
	| "apple/MobileCLIP-B-OpenCLIP"
