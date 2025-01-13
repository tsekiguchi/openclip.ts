import { apis } from "./env.js";
import { DEVICE_TYPES } from "./devices.js";

// TODO: Use the adapter from `env.backends.onnx.webgpu.adapter` to check for `shader-f16` support,
// when available in https://github.com/microsoft/onnxruntime/pull/19940.
// For more information, see https://github.com/microsoft/onnxruntime/pull/19857#issuecomment-1999984753

/**
 * Checks if WebGPU fp16 support is available in the current environment.
 */
export const isWebGpuFp16Supported: () => Promise<boolean> = (() => {
	let cachedResult: boolean;

	return async () => {
		if (cachedResult === undefined) {
			if (!apis.IS_WEBGPU_AVAILABLE) {
				cachedResult = false;
			} else {
				try {
					const adapter = await navigator.gpu.requestAdapter();
					if (!adapter) throw new Error("WebGPU adapter not detected");
				} catch (e) {
					console.error(e);
					cachedResult = false;
				}
			}
		}
		return cachedResult;
	};
})();

export const DATA_TYPES = Object.freeze({
	auto: "auto", // Auto-detect based on environment
	fp32: "fp32",
	fp16: "fp16",
	q8: "q8",
	int8: "int8",
	uint8: "uint8",
	q4: "q4",
	bnb4: "bnb4",
	q4f16: "q4f16", // fp16 model with int4 block weight quantization
});
export type DataType = keyof typeof DATA_TYPES;

// Exclude "auto" from DataType for valid keys
export type ValidDataType = Exclude<DataType, "auto">;

export const DEFAULT_DEVICE_DTYPE_MAPPING = Object.freeze({
	// NOTE: If not specified, will default to fp32
	[DEVICE_TYPES.wasm]: DATA_TYPES.q8,
});

// Define the suffix mapping using a Record type and readonly properties
export const DEFAULT_DTYPE_SUFFIX_MAPPING = Object.freeze({
  fp32: "",
  fp16: "_fp16",
  int8: "_int8",
  uint8: "_uint8",
  q8: "_quantized",
  q4: "_q4",
  q4f16: "_q4f16",
  bnb4: "_bnb4",
} as const); 

// Infer the mapping type from the object
export type DefaultDTypeSuffixMapping = typeof DEFAULT_DTYPE_SUFFIX_MAPPING;

// /** @type {Record<Exclude<DataType, "auto">, string>} */
// export const DEFAULT_DTYPE_SUFFIX_MAPPING = Object.freeze({
// 	[DATA_TYPES.fp32]: "",
// 	[DATA_TYPES.fp16]: "_fp16",
// 	[DATA_TYPES.int8]: "_int8",
// 	[DATA_TYPES.uint8]: "_uint8",
// 	[DATA_TYPES.q8]: "_quantized",
// 	[DATA_TYPES.q4]: "_q4",
// 	[DATA_TYPES.q4f16]: "_q4f16",
// 	[DATA_TYPES.bnb4]: "_bnb4",
// });
