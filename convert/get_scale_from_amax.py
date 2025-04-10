import json
import os
import re
from typing import Any

import torch
from safetensors.torch import save_file


def _remap_key(key_dict: dict[str, Any]):
    # renaming the module to match HF modeling
    mappig = {
        "ffn.shared_experts.w1": "mlp.shared_experts.gate_proj",
        "ffn.shared_experts.w2": "mlp.shared_experts.down_proj",
        "ffn.shared_experts.w3": "mlp.shared_experts.up_proj",
        "ffn.shared_experts": "mlp.shared_experts",
        "ffn.shared_experts": "mlp.shared_experts",
        "ffn.shared_experts": "mlp.shared_experts",
        "ffn.w1": "mlp.gate_proj",
        "ffn.w2": "mlp.down_proj",
        "ffn.w3": "mlp.up_proj",
        "head": "lm_head",
        "attn": "self_attn",
    }

    new_dict = {}
    for k, v in key_dict.items():
        new_key = k.replace("layers", "model.layers")

        for original_pattern, replace_pattern in mappig.items():
            new_key = new_key.replace(original_pattern, replace_pattern)

        # ffn.experts.xx.w1/w2/w3- > mlp.experts.xx.gate_proj/down_proj/up_proj
        new_key = re.sub(r"ffn\.experts\.(\d+)\.w1",
                         r"mlp.experts.\1.gate_proj", new_key)
        new_key = re.sub(r"ffn\.experts\.(\d+)\.w2",
                         r"mlp.experts.\1.down_proj", new_key)
        new_key = re.sub(r"ffn\.experts\.(\d+)\.w3", r"mlp.experts.\1.up_proj",
                         new_key)

        new_dict[new_key] = v

    key_dict.clear()
    key_dict.update(new_dict)


def load_and_preprocess_state_dict(modelopt_state_root, world_size=8):
    state_dict_list = []
    # load amax from nvfp4 state dict
    for rank in range(world_size):
        state_dict_list.append(
            torch.load(
                f"{modelopt_state_root}/amax_dict_rank{rank}-mp{world_size}.pt")
        )

    # calculate the max across all TP
    merged_state_dict = state_dict_list[0]
    for rank in range(world_size):
        for key, amax in state_dict_list[rank].items():
            if key in merged_state_dict.items():
                amax = torch.max(amax.to(0),
                                 merged_state_dict[key].to(0))  #amax.device))
            merged_state_dict[key] = amax.to(0)

    _remap_key(merged_state_dict)

    # set amax for modules to be fused and make sure they share the same input
    for key, amax in merged_state_dict.items():
        if "up_proj" in key:
            gate_proj_key = key.replace("up_proj", "gate_proj")
            if "weight_quantizer" in key:
                fused_amax = torch.max(amax, merged_state_dict[gate_proj_key])
                merged_state_dict[key] = fused_amax
                merged_state_dict[gate_proj_key] = fused_amax
            elif "input_quantizer" in key:
                assert amax == merged_state_dict[gate_proj_key]
            else:
                raise NotImplementedError

    return merged_state_dict


def process_quant_config(quant_config_path: str, save_path: str):
    with open(quant_config_path, "r") as f:
        quant_config = json.load(f)

    if "exclude_modules" in quant_config["quantization"]:
        exclude_dict = {
            k: None
            for k in quant_config["quantization"]["exclude_modules"]
        }
        _remap_key(exclude_dict)
        quant_config["quantization"]["exclude_modules"] = list(
            exclude_dict.keys())

    if "quantized_layers" in quant_config["quantization"]:
        _remap_key(quant_config["quantization"]["quantized_layers"])

    with open(save_path, "w") as f:
        json.dump(quant_config, f, indent=4)


def remove_keys_with_weight_quantizer(dictionary):
    new_dict = {
        k: v
        for k, v in dictionary.items() if "weight_quantizer" not in k
    }

    return new_dict


'''
renamed_state_dict = remove_keys_with_weight_quantizer(renamed_state_dict)
layer_1 = [x for x in renamed_state_dict.keys() if x.startswith('model.layers.1.')]
layer_1.sort()
for name in layer_1:
    print(name)
'''


def find_max_value(scales, renamed_state_dict):
    max_value = float('-inf')

    for key in scales:
        max_value = max(max_value, renamed_state_dict[key])

    return max_value


def get_scales_from_amax(start_layer, end_layer, renamed_state_dict, scale_dir):
    scales = {}
    for layer_idx in range(start_layer, end_layer):
        amax_per_layer = [
            x for x in renamed_state_dict.keys()
            if x.startswith(f'model.layers.{layer_idx}.')
        ]
        #print(amax_per_layer)
        down_proj_amax = [x for x in amax_per_layer if "down_proj" in x]
        #print(down_proj_amax)
        gate_up_amax = [
            x for x in amax_per_layer if "gate_proj" in x or "up_proj" in x
        ]
        #print(gate_up_amax)
        w2_scale = find_max_value(down_proj_amax, renamed_state_dict).cpu().to(
            torch.float32) / 448
        w3_w1_scale = find_max_value(gate_up_amax, renamed_state_dict).cpu().to(
            torch.float32) / 448
        key = f'model.layers.{layer_idx}.mlp.experts'
        scales.update({f'{key}.w3_w1_scale': w3_w1_scale})
        scales.update({f'{key}.w2_scale': w2_scale})

    save_file(scales, os.path.join(scale_dir, "input_scales.safetensors"))


import argparse

parser = argparse.ArgumentParser(description='从amax值获取scale')
parser.add_argument('--start_layer', type=int, required=True, help='起始层数')
parser.add_argument('--end_layer', type=int, required=True, help='结束层数')
parser.add_argument('--amax_dir', type=str, required=True, help='amax文件目录')
parser.add_argument('--scale_dir', type=str, required=True, help='scale文件保存目录')

args = parser.parse_args()
print(args.start_layer, args.end_layer, args.amax_dir, args.scale_dir)
#if not all([args.start_layer, args.end_layer, args.amax_dir, args.scale_dir]):
#    raise ValueError("必须提供所有参数: start_layer, end_layer, amax_dir, scale_dir")

if not os.path.exists(args.amax_dir):
    raise ValueError(f"amax文件目录不存在: {args.amax_dir}")
if not os.path.exists(args.scale_dir):
    os.makedirs(args.scale_dir)

start_layer = args.start_layer
end_layer = args.end_layer

process_quant_config(
    quant_config_path=os.path.join(args.amax_dir, "hf_quant_config.json"),
    save_path=os.path.join(args.scale_dir, "hf_quant_config.json"),
)

renamed_state_dict = load_and_preprocess_state_dict(
    modelopt_state_root=args.amax_dir, world_size=8)

get_scales_from_amax(start_layer=start_layer,
                     end_layer=end_layer,
                     renamed_state_dict=renamed_state_dict,
                     scale_dir=args.scale_dir)
