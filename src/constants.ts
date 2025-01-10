// Tuple type for RGB values
type RGBTuple = readonly [number, number, number];

interface ProcessConfig {
	mean: RGBTuple;
	std: RGBTuple;
	interpolation: "bicubic" | "bilinear";
	resizeMode: "squash" | "shortest";
}

interface ProcessConfigs {
	openAI: ProcessConfig;
	imageNet: ProcessConfig;
	inception: ProcessConfig;
	mobileCLIP: ProcessConfig;
}

export const ProcessConfigs: ProcessConfigs = {
	// OpenAI / OpenCLIP defaults
	openAI: {
		mean: [0.48145466, 0.4578275, 0.40821073],
		std: [0.26862954, 0.26130258, 0.27577711],
		interpolation: "bicubic",
		resizeMode: "shortest",
	},
	// CLIPA Defaults
	imageNet: {
		mean: [0.485, 0.456, 0.406],
		std: [0.229, 0.224, 0.225],
		interpolation: "bicubic",
		resizeMode: "shortest",
	},
	// SiGLIP defaults
	inception: {
		mean: [0.5, 0.5, 0.5] as RGBTuple,
		std: [0.5, 0.5, 0.5] as RGBTuple,
		interpolation: 'bicubic',
		resizeMode: 'squash'
	},
	// MobileClip
	mobileCLIP: {
		mean: [0, 0, 0],
		std: [1, 1, 1],
		interpolation: 'bilinear',
		resizeMode: 'shortest'
	},
} as const;

export const HF_FILES = {
	WEIGHTS_NAME: "open_clip_pytorch_model.bin",
	SAFE_WEIGHTS_NAME: "open_clip_model.safetensors",
	CONFIG_NAME: "open_clip_config.json",
} as const;
