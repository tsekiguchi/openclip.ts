import os from "os";
import path from "path";
import FS from "fs";
import fs from "fs/promises";

type VITB32Type = {
	openai: ModelConfig;
	laion400m_e31: ModelConfig;
	// laion400m_e32: ModelConfig;
	laion2b_e16: ModelConfig;
	laion2b_s34b_b79k: ModelConfig;
	// datacomp_xl_s13b_b90k: ModelConfig;
	// datacomp_m_s128m_b4k: ModelConfig;
	// commonpool_m_clip_s128m_b4k: ModelConfig;
	// commonpool_m_laion_s128m_b4k: ModelConfig;
	// commonpool_m_image_s128m_b4k: ModelConfig;
	// commonpool_m_text_s128m_b4k: ModelConfig;
	// commonpool_m_basic_s128m_b4k: ModelConfig;
	// commonpool_m_s128m_b4k: ModelConfig;
	// datacomp_s_s13m_b4k: ModelConfig;
	// commonpool_s_clip_s13m_b4k: ModelConfig;
	// commonpool_s_laion_s13m_b4k: ModelConfig;
	// commonpool_s_image_s13m_b4k: ModelConfig;
	// commonpool_s_text_s13m_b4k: ModelConfig;
	// commonpool_s_basic_s13m_b4k: ModelConfig;
	// commonpool_s_s13m_b4k: ModelConfig;
	// metaclip_400m: ModelConfig;
	// metaclip_fullcc: ModelConfig;
};

// Define the configuration interface
interface ModelConfig {
	url?: string;
	hf_hub: string;
	quick_gelu?: boolean;
}

// Define the VITB32 constant
const VITB32: VITB32Type = {
	openai: {
		url: "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
		hf_hub: "timm/vit_base_patch32_clip_224.openai/",
		quick_gelu: true,
	},
	laion400m_e31: {
		url: "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt",
		hf_hub: "timm/vit_base_patch32_clip_224.laion400m_e31/",
		quick_gelu: true,
	},
	laion2b_e16: {
		url: "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-laion2b_e16-af8dbd0c.pth",
		hf_hub: "timm/vit_base_patch32_clip_224.laion2b_e16/",
	},
	laion2b_s34b_b79k: {
		hf_hub: "laion/CLIP-ViT-B-32-laion2B-s34B-b79K/",
	},
} as const;

// Export as frozen object to prevent modifications
export default Object.freeze(VITB32);

export class PreTrainedModel {
	private directory = path.join(os.homedir(), ".modelscache");

	constructor() {
		this.directory = path.join(os.homedir(), ".modelscache");
		if (!FS.existsSync(this.directory)) FS.mkdirSync(this.directory);
	}

	// public getPretrainedModelFromHF(modelName: VITB32Type): Promise<void> {
	// 	// Get the hugging face hub for the model
	// 	// Check to see if the model has a SafeTensors alt,
	// 	// File ext .safetensors
	// }

	public async getPretrainedModelFromLocal(modelPath: string): Promise<void> {
		try {
			if (!FS.existsSync(modelPath)) {
				throw new Error("Model at path does not exist");
			}
			const stats = await fs.stat(modelPath);

			if (stats.isDirectory()) {
				throw new Error("Path is directory, must provide a file");
			}

			const ext = path.extname(modelPath);

			const validExtensions = ["bin", "safetensors"];
			if (!validExtensions.includes(ext)) {
				throw new Error('Invalid file type - must be .bin or .safetensors')
			}

			console.log('model path', modelPath)

		} catch (error) {
			console.error('Error gettingPretrainedModel: ', error)
		}
	}
}
