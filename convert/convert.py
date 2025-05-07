import shutil
import torch
from safetensors.torch import load_file, save_file, safe_open
import argparse
import json
import os
import tensorrt_llm
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help='HF checkpoint path')
    parser.add_argument('--output_dir', type=str, required=True, help='Save path')
    parser.add_argument('--act_scales', type=str, required=True, help='Activation scales exported from ModelOpt')
    parser.add_argument('--layer', type=int, default=61, help='number of layer')
    parser.add_argument('--parts', type=int, default=1, help='devide all safttensors into parts')
    parser.add_argument('--rank', type=int, default=0, help='which part to be quantize')
    args = parser.parse_args()
    return args

def fp8_bf16_int4(fp8_tensor, fp8_scale):
    group_size = 128
    blocked_tensor = fp8_tensor.view(fp8_tensor.shape[0] // 128, 128, fp8_tensor.shape[1] // 128, 128).to(torch.float32)
    dequant_tensor = (blocked_tensor * fp8_scale.unsqueeze(1).unsqueeze(3)).view(fp8_tensor.shape[0], fp8_tensor.shape[1] // group_size, group_size).to(torch.bfloat16).to(torch.float32)
    scale_tensor = torch.abs(dequant_tensor).max(dim=2).values/7
    quant_tensor = torch.clamp(torch.round((dequant_tensor / scale_tensor.unsqueeze(-1))), min=-8, max=7)
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
        processed_files[file_name] = safe_open(os.path.join(model_dir, file_name), "pt", device="cuda")

    args.layer = args.layer + 1
    part_layer = (args.layer + args.parts - 1) // args.parts
    start_layer = args.rank*part_layer
    end_layer = args.rank*part_layer+part_layer
    if end_layer > args.layer:
        end_layer = args.layer
    
    
    def get_tensor(name):
        if name not in weight_map:
            return None
        ff = weight_map[name]
        safetensors_loader = processed_files[ff]
        return safetensors_loader.get_tensor(name).cuda()
    
    def get_file_name(layer):
        rank = layer//part_layer
        return f"quantized_int4_rank{rank}.safetensors"

    new_safetensors = {}
    new_json = {}
    new_json['weight_map'] = {}
    new_json['metadata'] = {}
    for key in tqdm(list(weight_map.keys())):
        if "mlp.experts" in key and (key.endswith("weight") or key.endswith("weight_scale_inv")):
            if key.endswith("weight_scale_inv"):
                continue
            if args.rank == 0:
                layer = int(key.split(".")[2])
                new_json['weight_map'][key] = get_file_name(layer)
                new_json['weight_map'][key.replace("weight", "weight_scale_inv")] = get_file_name(layer)
            if int(key.split(".")[2]) < start_layer or int(key.split(".")[2]) >= end_layer:
                continue
            fp8_tensor = get_tensor(key)
            fp8_scale = get_tensor(key.replace("weight", "weight_scale_inv"))
            quant_tensor, scale_tensor = fp8_bf16_int4(fp8_tensor, fp8_scale)

            import tensorrt_llm
            packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
            packed_tensor = packer(quant_tensor.cpu().contiguous())
            new_safetensors.update({key: packed_tensor})
            new_safetensors.update({key.replace("weight", "weight_scale_inv"): scale_tensor.contiguous()})
        else:
            name = key.split(".")
            if args.rank == 0:
                if len(name) < 3 or not name[2].isdigit():
                    new_safetensors.update({key: get_tensor(key)})
                    new_json['weight_map'][key] = get_file_name(0)
                    continue
                
                file_name = get_file_name(int(name[2]))
                new_json['weight_map'][key] = file_name
                
            if len(name) < 3 or not name[2].isdigit() or (int(name[2]) < start_layer or int(name[2]) >= end_layer):
                continue
            new_safetensors.update({key: get_tensor(key)})
    
    if args.rank == 0:
        input_scales = safe_open(args.act_scales, "pt")
        for k in input_scales.keys():
            new_safetensors.update({k: input_scales.get_tensor(k)})
            new_json['weight_map'][k] = "input_scales.safetensors"

        file_name = get_file_name(start_layer)
        print(f'saving to {file_name}...')
        save_file(new_safetensors, os.path.join(output_dir, file_name))
        if args.rank == 0:
            new_json['metadata']['total_size'] = 376619902288
            with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:  
                json.dump(new_json, f)
            
            names = [
                "configuration_deepseek.py", "generation_config.json",
                "modeling_deepseek.py", "tokenizer.json", "tokenizer_config.json"
            ]
            for name in names:
                shutil.copy(os.path.join(model_dir, name), output_dir)
            shutil.copy(args.act_scales, output_dir)

        # config.json
        with open(os.path.join(model_dir, "config.json"), 'r') as file:
            config = json.load(file)
        del config['quantization_config']
        with open(os.path.join(output_dir, "quant_cfg.json"), 'w') as file:
            json.dump(config, file, indent=4)
        
        # quant_cfg.json
        attn_names = ["fused_a", "q_b_proj", "kv_b_proj", "o_proj"]
        mlp_names = ["gate_up_proj", "down_proj"]
        d = {}
        d["quant_algo"] = "MIXED_PRECISION"
        d["kv_cache_quant_algo"] = None
        d["quantized_layers"] = {}
        for l in range(61):
            prefix = f"model.layers.{l}"
            for n1 in attn_names:
                d["quantized_layers"][f"{prefix}.self_attn.{n1}"] = {"quant_algo": "FP8_BLOCK_SCALES"}
            for n2 in mlp_names:
                d["quantized_layers"][f"{prefix}.mlp.shared_experts.{n2}"] = {"quant_algo": "FP8_BLOCK_SCALES"}
            if l < 3:
                for n3 in mlp_names:
                    d["quantized_layers"][f"{prefix}.mlp.{n3}"] = {"quant_algo": "FP8_BLOCK_SCALES"}
            else:
                d["quantized_layers"][f"{prefix}.mlp.experts"] = {"quant_algo": "W4A8_AWQ"}
        with open(os.path.join(output_dir, "quant_cfg.json"), 'w') as file:
            json.dump(d, file, indent=4)
        
        # hf_quant_config.json
        d = {}
        d['quantization'] = {}
        d['quantization']["quant_algo"] = "MIXED_PRECISION"
        d['quantization']["kv_cache_quant_algo"] = None
        with open(os.path.join(output_dir, "hf_quant_config.json"), 'w') as file:
            json.dump(d, file, indent=4)

if __name__ == "__main__":
    args = parse_args()
    main(args)
