/**
 * @file Helper module for using model configs. For more information, see the corresponding
 * [Python documentation](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoConfig).
 *
 * **Example:** Load an `AutoConfig`.
 *
 * ```javascript
 * import { AutoConfig } from '@huggingface/transformers';
 * const config = await AutoConfig.from_pretrained('bert-base-uncased');
 * console.log(config);
 * // PretrainedConfig {
 * //   "model_type": "bert",
 * //   "is_encoder_decoder": false,
 * //   "architectures": [
 * //       "BertForMaskedLM"
 * //   ],
 * //   "vocab_size": 30522
 * //   "num_attention_heads": 12,
 * //   "num_hidden_layers": 12,
 * //   "hidden_size": 768,
 * //   "max_position_embeddings": 512,
 * //   ...
 * // }
 * ```
 *
 * @module configs
 */

import { getModelJSON } from "./utils/hub.js";
import type { PretrainedOptions, ModelType, NormalizedConfig, PretrainedConfigJSON } from "./utils/interfaces.js";

/**
 * Helper class which is used to instantiate pretrained configs with the `from_pretrained` function.
 *
 * @example
 * const config = await AutoConfig.from_pretrained('Xenova/bert-base-uncased');
 */
export class AutoConfig {
	static async from_pretrained(modelType: ModelType) {
		return PretrainedConfig.from_pretrained(modelType);
	}
}

/**
 * Loads a config from the specified path.
 * @param pretrained_model_name_or_path The path to the config directory.
 * @param options Additional options for loading the config.
 * @returns A promise that resolves with information about the loaded config.
 */
async function loadConfig(
	pretrained_model_name_or_path: string,
	options: PretrainedOptions
): Promise<PretrainedConfigJSON> {
	const config = await getModelJSON(
		pretrained_model_name_or_path,
		"config.json",
		true,
		options
	);

	if (!config || typeof config !== "object") {
		throw new Error("Invalid configuration loaded.");
	}

	return config as PretrainedConfigJSON;
}

/**
 *
 * @param config
 * @returns The normalized configuration.
 */
function getNormalizedConfig(config: PretrainedConfigJSON): NormalizedConfig {
	const mapping = {} as Record<keyof PretrainedConfigJSON, string>

	let initNormalizedConfig: NormalizedConfig = {
		model_type: config.model_type,
	};

	switch (config.model_type) {
		// Sub-configs
		case "llava":
		case "paligemma":
		case "florence2":
		case "llava_onevision":
		case "idefics3": {
			const text_config = config.text_config as PretrainedConfigJSON;
			if (!text_config) throw new Error("text_config not found");
			initNormalizedConfig = getNormalizedConfig(text_config);
			break;
		}

		case "moondream1": {
			const phi_config = config.phi_config as PretrainedConfigJSON;
			if (!phi_config) throw new Error("phi_config not found");
			initNormalizedConfig = getNormalizedConfig(phi_config);
			break;
		}

		case "musicgen": {
			const decoder = config.decoder as PretrainedConfigJSON;
			if (!decoder) throw new Error("decoder not found");
			initNormalizedConfig = getNormalizedConfig(decoder);
			break;
		}

		case "multi_modality":
			const language_config = config.language_config as PretrainedConfigJSON;
			if (!language_config) throw new Error("language_config not found");
			initNormalizedConfig = getNormalizedConfig(language_config);
			break;

		// Decoder-only models
		case "gpt2":
		case "gptj":
		case "jais":
		case "codegen":
		case "gpt_bigcode": {
			mapping["num_heads"] = "n_head";
			mapping["num_layers"] = "n_layer";
			mapping["hidden_size"] = "n_embd";
			break;
		}

		case "gpt_neox":
		case "stablelm":
		case "opt":
		case "falcon": {
			mapping["num_heads"] = "num_attention_heads";
			mapping["num_layers"] = "num_hidden_layers";
			mapping["hidden_size"] = "hidden_size";
			break;
		}

		case "llama":
		case "olmo":
		case "olmo2":
		case "mobilellm":
		case "granite":
		case "cohere":
		case "mistral":
		case "starcoder2":
		case "qwen2":
		case "qwen2_vl":
		case "phi":
		case "phi3":
		case "phi3_v": {
			mapping["num_heads"] = "num_key_value_heads";
			mapping["num_layers"] = "num_hidden_layers";
			mapping["hidden_size"] = "hidden_size";
			mapping["num_attention_heads"] = "num_attention_heads";
			break;
		}

		case "gemma":
		case "gemma2": {
			mapping["num_heads"] = "num_key_value_heads";
			mapping["num_layers"] = "num_hidden_layers";
			mapping["dim_kv"] = "head_dim";
			break;
		}

		case "openelm": {
			mapping["num_heads"] = "num_kv_heads";
			mapping["num_layers"] = "num_transformer_layers";
			mapping["dim_kv"] = "head_dim";
			break;
		}

		case "gpt_neo":
		case "donut-swin": {
			mapping["num_heads"] = "num_heads";
			mapping["num_layers"] = "num_layers";
			mapping["hidden_size"] = "hidden_size";
			break;
		}

		case "bloom": {
			mapping["num_heads"] = "n_head";
			mapping["num_layers"] = "n_layer";
			mapping["hidden_size"] = "hidden_size";
			break;
		}

		case "mpt": {
			mapping["num_heads"] = "n_heads";
			mapping["num_layers"] = "n_layers";
			mapping["hidden_size"] = "d_model";
			break;
		}

		case "exaone": {
			mapping["num_heads"] = "num_key_value_heads";
			mapping["num_layers"] = "num_layers";
			mapping["dim_kv"] = "head_dim";
			mapping["num_attention_heads"] = "num_attention_heads";
			break;
		}

		// Encoder-decoder models
		case "t5":
		case "mt5":
		case "longt5": {
			mapping["num_decoder_layers"] = "num_decoder_layers";
			mapping["num_decoder_heads"] = "num_heads";
			mapping["decoder_dim_kv"] = "d_kv";
			mapping["num_encoder_layers"] = "num_layers";
			mapping["num_encoder_heads"] = "num_heads";
			mapping["encoder_dim_kv"] = "d_kv";
			break;
		}

		case "bart":
		case "mbart":
		case "marian":
		case "whisper":
		case "m2m_100":
		case "blenderbot":
		case "blenderbot-small":
		case "florence2_language": {
			mapping["num_decoder_layers"] = "decoder_layers";
			mapping["num_decoder_heads"] = "decoder_attention_heads";
			mapping["decoder_hidden_size"] = "d_model";
			mapping["num_encoder_layers"] = "encoder_layers";
			mapping["num_encoder_heads"] = "encoder_attention_heads";
			mapping["encoder_hidden_size"] = "d_model";
			break;
		}

		case "speecht5": {
			mapping["num_decoder_layers"] = "decoder_layers";
			mapping["num_decoder_heads"] = "decoder_attention_heads";
			mapping["decoder_hidden_size"] = "hidden_size";
			mapping["num_encoder_layers"] = "encoder_layers";
			mapping["num_encoder_heads"] = "encoder_attention_heads";
			mapping["encoder_hidden_size"] = "hidden_size";
			break;
		}

		case "trocr": {
			mapping["num_encoder_layers"] = mapping["num_decoder_layers"] =
				"decoder_layers";
			mapping["num_encoder_heads"] = mapping["num_decoder_heads"] =
				"decoder_attention_heads";
			mapping["encoder_hidden_size"] = mapping["decoder_hidden_size"] =
				"d_model";
			break;
		}

		case "musicgen_decoder":
		case "moonshine": {
			mapping["num_encoder_layers"] = mapping["num_decoder_layers"] =
				"num_hidden_layers";
			mapping["num_encoder_heads"] = mapping["num_decoder_heads"] =
				"num_attention_heads";
			mapping["encoder_hidden_size"] = mapping["decoder_hidden_size"] =
				"hidden_size";
			break;
		}

		case "vision-encoder-decoder":
			const decoder = config.decoder as PretrainedConfigJSON;
			if (!decoder) throw new Error("decoder not found");
			const decoderConfig = getNormalizedConfig(decoder);

			const result: NormalizedConfig = {
				model_type: config.model_type,
				is_encoder_decoder: config.is_encoder_decoder,
			};

			const add_encoder_pkv = "num_decoder_layers" in decoderConfig;

			if (add_encoder_pkv) {
				Object.assign(result, {
					num_decoder_layers: decoderConfig.num_decoder_layers,
					num_decoder_heads: decoderConfig.num_decoder_heads,
					decoder_hidden_size: decoderConfig.decoder_hidden_size,
					num_encoder_layers: decoderConfig.num_encoder_layers,
					num_encoder_heads: decoderConfig.num_encoder_heads,
					encoder_hidden_size: decoderConfig.encoder_hidden_size,
				});
			} else {
				// Decoder is a decoder-only model
				Object.assign(result, {
					num_layers: decoderConfig.num_layers,
					num_heads: decoderConfig.num_heads,
					hidden_size: decoderConfig.hidden_size,
				});
			}
			return result;
	}

	// NOTE: If `num_attention_heads` is not set, it is assumed to be equal to `num_heads`

	const normalized_config: NormalizedConfig = Object.assign(initNormalizedConfig, {
		model_type: config.model_type,
		multi_query: config.multi_query,
		is_encoder_decoder: config.is_encoder_decoder,
	});

	Object.entries(mapping).forEach(([normalizedKey, originalKey]) => {
		normalized_config[normalizedKey] = config[originalKey as keyof PretrainedConfigJSON]
	})

	return normalized_config;
}

/**
 * Base class for all configuration classes.
 */
export class PretrainedConfig {
	normalized_config: NormalizedConfig;
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

	is_encoder_decoder?: boolean;

	encoder_hidden_size?: number;
	num_encoder_layers?: number;
	num_encoder_heads?: number;
	encoder_dim_kv?: number;

	decoder_hidden_size?: number;
	num_decoder_layers?: number;
	num_decoder_heads?: number;
	decoder_dim_kv?: number;

	constructor(configJSON: PretrainedConfigJSON) {
		Object.assign(this, configJSON);
		this.model_type = configJSON.model_type
		this.normalized_config = getNormalizedConfig(configJSON);
	}

	/**
	 * Loads a pre-trained configuration from a given path.
	 */
	static async from_pretrained(
		pretrained_model_name_or_path: ModelType,
		options: PretrainedOptions = {}
	): Promise<PretrainedConfig> {
		const configJSON = await loadConfig(pretrained_model_name_or_path, options);
		return new this(configJSON);
	}
}
