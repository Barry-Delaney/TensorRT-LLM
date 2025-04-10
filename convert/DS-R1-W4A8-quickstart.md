
# TensorRT-LLM DeepSeek-R1 W4A8 guidance

## Prerequisites

* Checkout this specific branch `user/barry/dev`
* Using Hopper GPUs and make sure you have enough VRAM
* Have the [DeepSeek-R1 weights](https://huggingface.co/deepseek-ai/DeepSeek-R1) at your local and reserve enough disk space for the quantized checkpoint (~351GB)

## Checkpoint preparation

Please make sure you have TensorRT-LLM installed before preparing the W4A8 checkpoint.

### Checkpoint format

In this branch, we are employing W(INT)4-A(FP)8 for the MoE layers in DeepSeek-R1 and preserving the original accuracy for the reset layers. Per-tensor and per-group (1x128) quantiztaion is used for activation and weights respectively.

For a W4A8 MoE layer with `num_expert = E`, `inter_size = N` and `hidden_size = K`, the following components are required in the layer:

| Name | Dtype | Shape | Layout |
|:-:|:-:|:-:|:-:|
| `{LAYER_NAME}.w3_w1_weight` | INT4x2 | [E, N * 2, K / 2] | Packed INT4 |
| `{LAYER_NAME}.w2_weight` | INT4x2 | [E, K, N / 2] | Packed INT4 |
| `{LAYER_NAME}.fc31_weight_scale` | FP/BF16 | [E, K / 512, N * 8] | Interleaved |
| `{LAYER_NAME}.fc2_weight_scale` | FP/BF16 | [E, N / 512, K * 4] | Interleaved |
| `{LAYER_NAME}.fc31_act_scale` | FP/BF16 | [1, K] | - |
| `{LAYER_NAME}.fc2_act_scale` | FP/BF16 | [1, N] | - |
| `{LAYER_NAME}.fc31_alpha` | FP32 | [E, 1] | - |
| `{LAYER_NAME}.fc2_alpha` | FP32 | [E, 1] | - |

From the shapes you can tell that:
* Both `N` and `K` is required to be multiple of 512, otherwise you need padding
* The OP actually takes per-channel activation scale and per-expert alpha, but here we are filling with the same values across channels and experts

### Activation calibration

For generating the per-tensor BF16->FP8 scale, we use [ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer/) here.

To save user's time, we have also uploaded the converted [`input_scales.safetensors`](./input_scakes.safetensors) with this branch. This file is generated with the following commands:

```bash
PATH_OF_DEEPSEEK_R1=/llm-models/DeepSeek-R1/DeepSeek-R1

# Install ModelOpt from source
git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer/ && cd modelopt
pip install "nvidia-modelopt[all]" -U --extra-index-url https://pypi.nvidia.com

# Clone DeepSeek-V3 (base model of R1) Github repository for FP8 inference,
git clone https://github.com/deepseek-ai/DeepSeek-V3.git && cd DeepSeek-V3 && git checkout 1398800

# Convert the HF checkpoint to a specific format for Deepseek
python inference/convert.py --hf-ckpt-path $PATH_OF_DEEPSEEK_R1 --save-path ds_r1 --n-experts 256 --model-parallel 8 && cd ..

# Do per-tensor fp8 calibration
torchrun --nproc-per-node 8 --master_port=12346 ptq.py --model_path DeepSeek-V3/ds_r1 --config DeepSeek-V3/inference/configs/config_671B.json --quant_cfg FP8_DEFAULT_CFG --output_path ds_r1_fp8_per_tensor_calibration && cd ../..

# Get input scale
python convert/get_scale_from_amax.py --start_layer 0 --end_layer 61 --amax_dir ds_r1_fp8_per_tensor_calibration --scale_dir convert
```

### Weight quantization

The demo script uses RTN (Round-To-Nearest) on the original FP8 block-wise quantized weights.

```bash
# Use the checkpoint in this branch or user's own file
mkdir convert/ckpt && cp convert/input_scales.safetensors convert/ckpt/
python convert/int4.py --model_dir PATH_OF_DEEPSEEK_R1 --save_dir convert/ckpt
```

### Assembeling

To get this checkpoint recognized by torch, tokenizers and configurations should also be copied into the folder.

```bash
cp PATH_OF_DEEPSEEK_R1/config* convert/ckpt/
cp PATH_OF_DEEPSEEK_R1/tokenizer* convert/ckpt/
cp PATH_OF_DEEPSEEK_R1/generation_config.json convert/ckpt/
```

## Quickstart

This command can be used for quick verification:

```bash
# The test is workable with 4xH200s or 8xH100s
python examples/pytorch/quickstart_advanced.py --model_dir convert/ckpt/ --tp_size 4 --moe_ep_size 4 --moe_tp_size 1
```

And the output will be:
```bash
[0] Prompt: 'Hello, my name is', Generated text: '\n\n# Dr. Kaitlyn Buss\n\n## Professional information\n\n- Designation\n\n-  Veterinarian, DVM\n\n- About Vet\n\nHi! My name is Dr. Kaitlyn Buss. I am a small animal veterinarian in the state of Michigan. I have a special interest in internal medicine'
[1] Prompt: 'The president of the United States is', Generated text: ' the head of state and head of government of the United States, indirectly elected to a four-year term by the American people through the Electoral College. The officeholder leads the executive branch of the federal government and is the commander-in-chief of the United States Armed Forces.\n\nSince the office was established in 1789, '
[2] Prompt: 'The capital of France is', Generated text: ' Paris. Paris is located in the northern part of the country, along the Seine River. It is one of the most famous and visited cities in the world, known for its rich history, culture, art, fashion, and cuisine. Paris is also a major global center for business, education, and diplomacy. The city'
[3] Prompt: 'The future of AI is', Generated text: " here, and it's changing the way we work. With the rise of artificial intelligence, businesses are finding new and innovative ways to streamline their operations, increase efficiency, and improve customer experiences. From chatbots to predictive analytics, AI is transforming industries across the board. But what does this mean for the workforce? Will AI replace"
```

## Trouble shooting

Unittest is provided at [test.py](../test.py).
