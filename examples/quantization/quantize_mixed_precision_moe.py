# autoflake: skip_file
import argparse
import glob
import json
import os
import re
import shutil

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

import tensorrt_llm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        type=str,
                        required=True,
                        help='HF checkpoint path')
    parser.add_argument('--output_dir',
                        type=str,
                        required=True,
                        help='Save path')
    parser.add_argument(
        '--act_scales',
        type=str,
        required=True,
        help=
        'ModelOpt calibrated checkpoint dir or extracted safetensors for activation scales'
    )
    parser.add_argument('--parts',
                        type=int,
                        default=1,
                        help='devide all safetensors into parts')
    parser.add_argument('--rank',
                        type=int,
                        default=0,
                        help='which part to be quantize')
    args = parser.parse_args()
    return args


def load_and_preprocess_state_dict(modelopt_state_root, world_size=8):
    state_dict_list = []
    # Load every amax rank dump present, regardless of the mp{N} suffix. The
    # previous code looked up a hardcoded mp{world_size} (default 8), so a
    # calibration run with any other world size silently produced no scales.
    amax_files = sorted(
        glob.glob(os.path.join(modelopt_state_root, "amax_dict_rank*-mp*.pt")))
    for amax_file in amax_files:
        state_dict_list.append(torch.load(amax_file, map_location="cuda:0"))

    if not state_dict_list:
        print("ERROR: No amax_dict_rank*-mp*.pt files found in "
              f"{modelopt_state_root}")
        return {}

    # calculate the max across all loaded TP ranks. Iterate over the loaded
    # state dicts directly (not range(world_size)) so a missing rank file does
    # not cause an IndexError, and use `key in merged_state_dict` for the
    # membership test (`key in dict.items()` compares against (k, v) tuples and
    # is always False, which silently disabled the cross-rank max).
    merged_state_dict = {}
    for state_dict in state_dict_list:
        for key, amax in state_dict.items():
            amax = amax.to(0)
            if key in merged_state_dict:
                amax = torch.max(amax, merged_state_dict[key])
            merged_state_dict[key] = amax

    mapping = {
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
    for k, v in merged_state_dict.items():
        new_key = k.replace("layers", "model.layers")
        for original_pattern, replace_pattern in mapping.items():
            new_key = new_key.replace(original_pattern, replace_pattern)
        # ffn.experts.xx.w1/w2/w3- > mlp.experts.xx.gate_proj/down_proj/up_proj
        new_key = re.sub(r"ffn\.experts\.(\d+)\.w1",
                         r"mlp.experts.\1.gate_proj", new_key)
        new_key = re.sub(r"ffn\.experts\.(\d+)\.w2",
                         r"mlp.experts.\1.down_proj", new_key)
        new_key = re.sub(r"ffn\.experts\.(\d+)\.w3", r"mlp.experts.\1.up_proj",
                         new_key)
        new_dict[new_key] = v

    merged_state_dict.clear()
    merged_state_dict.update(new_dict)

    # set amax for modules to be fused and make sure they share the same input
    for key, amax in merged_state_dict.items():
        if "up_proj" in key:
            gate_proj_key = key.replace("up_proj", "gate_proj")
            if "weight_quantizer" in key:
                fused_amax = torch.max(amax, merged_state_dict[gate_proj_key])
                merged_state_dict[key] = fused_amax
                merged_state_dict[gate_proj_key] = fused_amax
            elif "input_quantizer" in key:
                # gate_proj and up_proj are fused and consume the same input, so
                # they must share a single input scale. Force both to the max
                # rather than asserting bit-exact equality: calibration can leave
                # tiny per-module differences, and `assert tensor == tensor`
                # would raise on a non-scalar amax.
                fused_amax = torch.max(amax, merged_state_dict[gate_proj_key])
                merged_state_dict[key] = fused_amax
                merged_state_dict[gate_proj_key] = fused_amax
            else:
                raise NotImplementedError

    return merged_state_dict


def get_scales_from_amax(start_layer, end_layer, renamed_state_dict):
    weight_name_dict = {"gate_proj": 1, "down_proj": 2, "up_proj": 3}
    scales = {}
    for layer_idx in range(start_layer, end_layer):
        amax_keys_per_layer = [
            x for x in renamed_state_dict.keys()
            if (x.startswith(f'model.layers.{layer_idx}.mlp.experts.')
                and x.endswith(".input_quantizer._amax"))
        ]
        for k in amax_keys_per_layer:
            expert_idx = int(k.split('.')[5])
            weight_idx = weight_name_dict[k.split('.')[6]]
            val = renamed_state_dict[k]
            scales[
                f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.w{weight_idx}.input_scale'] = val.unsqueeze(
                    0) / 448

    return scales


def quantize_fp8_block_scale_to_int4(fp8_tensor, fp8_scale):
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


def main(args):
    model_dir = args.model_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(args.rank % num_gpus)

    model_index_file = os.path.join(model_dir, "model.safetensors.index.json")
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
        processed_files[file_name] = safe_open(os.path.join(
            model_dir, file_name),
                                               "pt",
                                               device="cuda")

    with open(os.path.join(model_dir, "config.json"), 'r') as file:
        config = json.load(file)

    num_layer = config['num_hidden_layers']
    # Include the MTP / multi-token-prediction layers
    # (model.layers.{num_layer .. num_layer+num_nextn-1}) in the processing
    # range so their weights are written (otherwise rank 0 registers them in the
    # index but no rank writes them -> dangling). The MTP MoE is kept as the
    # source FP8 block scale rather than requantized to W4A8: ModelOpt does not
    # calibrate the MTP layer, so there are no W4A8 activation scales for it.
    num_nextn = config.get('num_nextn_predict_layers', 0)
    num_total_layer = num_layer + num_nextn
    part_layer = (num_total_layer + args.parts - 1) // args.parts
    start_layer = args.rank * part_layer
    end_layer = min(num_total_layer, args.rank * part_layer + part_layer)

    def get_tensor(name):
        if name not in weight_map:
            return None
        ff = weight_map[name]
        safetensors_loader = processed_files[ff]
        return safetensors_loader.get_tensor(name).cuda()

    def get_file_name(layer):
        rank = layer // part_layer
        return "model-%05d-of-%05d.safetensors" % (rank, args.parts)

    new_safetensors = {}
    new_json = {}
    new_json['weight_map'] = {}
    new_json['metadata'] = {}
    for key in tqdm(list(weight_map.keys())):
        key_parts = key.split(".")
        key_layer = int(key_parts[2]) if (len(key_parts) > 2
                                          and key_parts[2].isdigit()) else None
        # MTP-layer experts (layer index >= num_layer) are NOT requantized to
        # int4; they fall through to the copy path below and are kept as the
        # source FP8 block scale (no W4A8 activation scales exist for them).
        is_mtp_layer = key_layer is not None and key_layer >= num_layer
        if (not is_mtp_layer) and "mlp.experts" in key and (
                key.endswith("weight") or key.endswith("weight_scale_inv")):
            if key.endswith("weight_scale_inv"):
                continue
            if args.rank == 0:
                layer = int(key.split(".")[2])
                new_json['weight_map'][key] = get_file_name(layer)
                new_json['weight_map'][key.replace(
                    "weight", "weight_scale_inv")] = get_file_name(layer)
                # In the amax-dir (ModelOpt) path the per-expert activation
                # input_scale is generated by get_scales_from_amax and written
                # into the shard, but was never registered in the index, so the
                # W4A8 input scales were invisible to index-based loaders.
                # Register them here, mirroring the weight registration above.
                if os.path.isdir(args.act_scales):
                    proj = key.split(".")[-2]  # gate_proj / up_proj / down_proj
                    widx = {"gate_proj": 1, "down_proj": 2, "up_proj": 3}[proj]
                    new_json['weight_map'][key.replace(
                        f"{proj}.weight",
                        f"w{widx}.input_scale")] = get_file_name(layer)
            if int(key.split(".")[2]) < start_layer or int(
                    key.split(".")[2]) >= end_layer:
                continue
            fp8_tensor = get_tensor(key)
            fp8_scale = get_tensor(key.replace("weight", "weight_scale_inv"))
            quant_tensor, scale_tensor = quantize_fp8_block_scale_to_int4(
                fp8_tensor, fp8_scale)

            packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
            packed_tensor = packer(quant_tensor.cpu().contiguous())
            new_safetensors.update({key: packed_tensor})
            new_safetensors.update({
                key.replace("weight", "weight_scale_inv"):
                scale_tensor.contiguous()
            })
        else:
            name = key.split(".")
            if args.rank == 0:
                if len(name) < 3 or not name[2].isdigit():
                    new_safetensors.update({key: get_tensor(key)})
                    new_json['weight_map'][key] = get_file_name(0)
                    continue

                file_name = get_file_name(int(name[2]))
                new_json['weight_map'][key] = file_name

            if len(name) < 3 or not name[2].isdigit() or (int(
                    name[2]) < start_layer or int(name[2]) >= end_layer):
                continue
            new_safetensors.update({key: get_tensor(key)})

    # Process activation scales for all ranks
    if os.path.isdir(args.act_scales):
        # Extract activation scales
        renamed_state_dict = load_and_preprocess_state_dict(
            modelopt_state_root=args.act_scales, world_size=8)
        scales = get_scales_from_amax(start_layer=start_layer,
                                      end_layer=end_layer,
                                      renamed_state_dict=renamed_state_dict)
        new_safetensors.update(scales)

    if args.rank == 0:
        if not os.path.isdir(args.act_scales):
            input_scales = safe_open(args.act_scales, "pt")
            for k in input_scales.keys():
                new_safetensors.update({k: input_scales.get_tensor(k)})
                new_json['weight_map'][k] = args.act_scales.split("/")[-1]

        file_name = get_file_name(start_layer)
        print(f'saving to {file_name}...')
        save_file(new_safetensors, os.path.join(output_dir, file_name))
        with open(os.path.join(output_dir, "model.safetensors.index.json"),
                  "w") as f:
            json.dump(new_json, f)

        names = [
            "configuration_deepseek.py", "generation_config.json",
            "modeling_deepseek.py", "tokenizer.json", "tokenizer_config.json"
        ]
        for name in names:
            src = os.path.join(model_dir, name)
            if os.path.exists(src):
                shutil.copy(src, output_dir)
            else:
                print(f"WARNING: aux file not found, skipping: {src}")
        if os.path.isdir(args.act_scales):
            shutil.copytree(args.act_scales, output_dir, dirs_exist_ok=True)
        else:
            shutil.copy(args.act_scales, output_dir)

        # config.json
        config.pop('quantization_config', None)
        with open(os.path.join(output_dir, "config.json"), 'w') as file:
            json.dump(config, file, indent=4)

        # quant_cfg.json
        attn_names = ["fused_a", "q_b_proj", "kv_b_proj", "o_proj"]
        mlp_names = ["gate_up_proj", "down_proj"]
        fp8_block_scale = {"quant_algo": "FP8_BLOCK_SCALES"}
        w4a8_awq = {"quant_algo": "W4A8_AWQ"}
        quant_cfg = {}
        quant_cfg["quant_algo"] = "MIXED_PRECISION"
        quant_cfg["kv_cache_quant_algo"] = None
        quant_cfg["quantized_layers"] = {}
        first_k_dense = config.get('first_k_dense_replace', 3)
        for l in range(num_total_layer):
            prefix = f"model.layers.{l}"
            is_mtp_layer = l >= num_layer
            for n1 in attn_names:
                quant_cfg["quantized_layers"][
                    f"{prefix}.self_attn.{n1}"] = fp8_block_scale
            for n2 in mlp_names:
                quant_cfg["quantized_layers"][
                    f"{prefix}.mlp.shared_experts.{n2}"] = fp8_block_scale
            if l < first_k_dense:
                for n3 in mlp_names:
                    quant_cfg["quantized_layers"][
                        f"{prefix}.mlp.{n3}"] = fp8_block_scale
            elif is_mtp_layer:
                # MTP MoE is kept as FP8 block scale (not W4A8), since ModelOpt
                # does not calibrate the MTP layer's activations.
                quant_cfg["quantized_layers"][
                    f"{prefix}.mlp.experts"] = fp8_block_scale
            else:
                quant_cfg["quantized_layers"][
                    f"{prefix}.mlp.experts"] = w4a8_awq
        with open(os.path.join(output_dir, "quant_cfg.json"), 'w') as file:
            json.dump(quant_cfg, file, indent=4)

        # hf_quant_config.json
        hf_quant_config = {}
        hf_quant_config['quantization'] = {}
        hf_quant_config['quantization']["quant_algo"] = "MIXED_PRECISION"
        hf_quant_config['quantization']["kv_cache_quant_algo"] = None
        with open(os.path.join(output_dir, "hf_quant_config.json"),
                  'w') as file:
            json.dump(hf_quant_config, file, indent=4)
    else:
        file_name = get_file_name(start_layer)
        print(f'saving to {file_name}...')
        save_file(new_safetensors, os.path.join(output_dir, file_name))


if __name__ == "__main__":
    args = parse_args()
    main(args)
