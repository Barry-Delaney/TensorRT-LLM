import argparse
import json
import os

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        type=str,
                        required=True,
                        help='checkpoint path')
    parser.add_argument('--save_path',
                        type=str,
                        required=True,
                        help='save path')
    args = parser.parse_args()
    return args


def fp8_bf16_int4(fp8_tensor, fp8_scale):
    group_size = 128
    blocked_tensor = fp8_tensor.view(fp8_tensor.shape[0] // 128, 128,
                                     fp8_tensor.shape[1] // 128,
                                     128).to(torch.float32)
    dequant_tensor = (blocked_tensor *
                      fp8_scale.unsqueeze(1).unsqueeze(3)).view(
                          fp8_tensor.shape[0],
                          fp8_tensor.shape[1] // group_size,
                          group_size).to(torch.bfloat16).to(torch.float32)
    scale_tensor = torch.abs(dequant_tensor).max(dim=2).values / 7
    quant_tensor = torch.clamp(torch.round(
        (dequant_tensor / scale_tensor.unsqueeze(-1))),
                               min=-8,
                               max=7)
    quant_tensor = quant_tensor.to(torch.int8)

    return quant_tensor.view(fp8_tensor.shape), scale_tensor


def fp8_bf16_fp8(fp8_tensor, fp8_scale):
    blocked_tensor = fp8_tensor.view(fp8_tensor.shape[0] // 128, 128,
                                     fp8_tensor.shape[1] // 128,
                                     128).to(torch.float32)
    dequant_tensor = (blocked_tensor *
                      fp8_scale.unsqueeze(1).unsqueeze(3)).view(
                          fp8_tensor.shape).to(torch.bfloat16).to(torch.float32)
    scale_tensor = torch.abs(dequant_tensor).max() / 448
    quant_tensor = dequant_tensor / scale_tensor

    return quant_tensor, scale_tensor


def main(args):
    path = args.model_dir
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    model_index_file = os.path.join(path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
        weight_map = model_index["weight_map"]

    processed_files = {}
    for tensor_name in list(weight_map.keys()):
        if tensor_name not in weight_map:
            continue
        file_name = weight_map[tensor_name]
        if file_name in processed_files:
            continue
        processed_files[file_name] = safe_open(os.path.join(path, file_name),
                                               "pt",
                                               device="cuda")

    def get_tensor(name):
        if name not in weight_map:
            return None
        ff = weight_map[name]
        safetensors_loader = processed_files[ff]
        return safetensors_loader.get_tensor(name).cuda()

    new_safetensors = {}
    new_json = {}
    new_json['weight_map'] = {}
    new_json['metadata'] = {}
    file_name = "quantized_int4.safetensors"
    for key in tqdm(list(weight_map.keys())):
        if "mlp.experts" in key and (key.endswith("weight")
                                     or key.endswith("weight_scale_inv")):
            if key.endswith("weight_scale_inv"):
                continue
            fp8_tensor = get_tensor(key)
            # if key.replace("weight", "weight_scale_inv") not in tensors.keys():
            #     continue
            fp8_scale = get_tensor(key.replace("weight", "weight_scale_inv"))
            quant_tensor, scale_tensor = fp8_bf16_int4(fp8_tensor, fp8_scale)

            packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
            packed_tensor = packer(quant_tensor.T.cpu().contiguous()).T
            new_safetensors.update({key: packed_tensor.contiguous()})
            new_safetensors.update({
                key.replace("weight", "weight_scale_inv"):
                scale_tensor.contiguous()
            })
            new_json['weight_map'][key] = file_name
            new_json['weight_map'][key.replace("weight",
                                               "weight_scale_inv")] = file_name

            print(key, packed_tensor.shape, scale_tensor.shape)
        else:
            new_safetensors.update({key: get_tensor(key)})
            new_json['weight_map'][key] = file_name

    input_scales = safe_open(
        os.path.join(save_path, "input_scales.safetensors"), "pt")
    for k in input_scales.keys():
        new_safetensors.update({k: input_scales.get_tensor(k)})
        new_json['weight_map'][k] = "input_scales.safetensors"

    print(f'saving to {file_name}...')
    new_json['metadata']['total_size'] = 1369062772000
    save_file(new_safetensors, os.path.join(save_path, file_name))
    with open(os.path.join(save_path, "model.safetensors.index.json"),
              "w") as f:
        json.dump(new_json, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
