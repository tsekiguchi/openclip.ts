export interface MobileCLIPConfig {
	embed_dim: number,
	vision_cfg: {
		timm_model_name: string,
		timm_model_pretrained: boolean,
		timm_pool: string,
		timm_proj: null,
		timm_drop: number,
		timm_drop_path: number,
		image_size: number
	},
	text_cfg: {
		context_length: number,
		vocab_size: number,
		width: number,
		heads: number,
		layers: number,
		no_causal_mask: boolean,
		hf_tokenizer_name?: string
	},
	custom_text: boolean
}

export const MobileCLIPBConfig: MobileCLIPConfig = {
	"embed_dim": 512,
	"vision_cfg": {
			"timm_model_name": "vit_base_mci_224",
			"timm_model_pretrained": false,
			"timm_pool": "token",
			"timm_proj": null,
			"timm_drop": 0.0,
			"timm_drop_path": 0.0,
			"image_size": 224
	},
	"text_cfg": {
			"context_length": 77,
			"vocab_size": 49408,
			"width": 512,
			"heads": 8,
			"layers": 12,
			"no_causal_mask": false
	},
	"custom_text": true
}

export const MobileCLIPS1Config: MobileCLIPConfig = {
	"embed_dim": 512,
	"vision_cfg": {
			"timm_model_name": "fastvit_mci1",
			"timm_model_pretrained": false,
			"timm_pool": "avg",
			"timm_proj": null,
			"timm_drop": 0.0,
			"timm_drop_path": 0.0,
			"image_size": 256
	},
	"text_cfg": {
			"context_length": 77,
			"vocab_size": 49408,
			"width": 512,
			"heads": 8,
			"layers": 12,
			"no_causal_mask": true
	},
	"custom_text": true
}

export const MobileCLIPS2Config: MobileCLIPConfig = {
	"embed_dim": 512,
	"vision_cfg": {
			"timm_model_name": "fastvit_mci2",
			"timm_model_pretrained": false,
			"timm_pool": "avg",
			"timm_proj": null,
			"timm_drop": 0.0,
			"timm_drop_path": 0.0,
			"image_size": 256
	},
	"text_cfg": {
			"context_length": 77,
			"vocab_size": 49408,
			"width": 512,
			"heads": 8,
			"layers": 12,
			"no_causal_mask": true
	},
	"custom_text": true
}

