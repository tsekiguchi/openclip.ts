import { PreTrainedModel } from "./pretrained.js";
import { SimpleTokenizer } from "./tokenizers/SimpleTokenizer.js";

// const preTrainedModel = new PreTrainedModel()

const localPath = '/Users/timsekiguchi/models/CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.safetensors'

// preTrainedModel.getPretrainedModelFromLocal(localPath);

const tokenizer = new SimpleTokenizer();

const tokens = tokenizer.encode("Hello my lady")

console.dir(tokens)