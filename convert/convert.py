import json

from safetensors import safe_open


def load_safetensors_keys(file_path):
    f = safe_open(file_path, framework="pt", device=0)
    return f.keys()


def create_index_json(keys, path):
    d = {}
    with open(path, 'r') as file:
        data = json.load(file)
        for k in keys:
            data['weight_map'][k] = "input_scales.safetensors"
        d = data

    with open(path + '_new', 'w') as file:
        json.dump(d, file)


def main():
    safetensors_path = "/workspace/projects/mina/w4_r1/input_scales.safetensors"
    output_json_path = "/workspace/projects/mina/w4_r1/model.safetensors.index.json"

    keys = load_safetensors_keys(safetensors_path)
    create_index_json(keys, output_json_path)


if __name__ == "__main__":
    main()

# import json
# from safetensors import safe_open

# def main():
#     output_json_path = "/workspace/projects/mina/w4_r1/model.safetensors.index.json"
#     with open(output_json_path, 'r') as file:
#         data = json.load(file)
#         for k, v in data['weight_map'].items():
#             print(k, v)
#             f = safe_open("/workspace/projects/mina/w4_r1/" + v, "pt")
#             print(f.get_tensor(k).shape, f.get_tensor(k).shape)

# if __name__ == "__main__":
#     main()
