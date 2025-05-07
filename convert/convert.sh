#!/bin/bash
HF_MODEL_DIR=/models/DeepSeek-R1/
OUTPUT_DIR=/workspace/projects/ckpt/
ACT_SCALES=convert/input_scales.safetensors

if [ ! -d "convert/convert_logs" ]; then
    mkdir convert/convert_logs
fi

pids=()
for i in 0 1 2 3 4 5 6 7
do
    python convert/convert.py --model_dir $HF_MODEL_DIR --output_dir $OUTPUT_DIR --act_scales $ACT_SCALES --layer 61 --parts 9 --rank $i > convert/convert_logs/log_$i 2>&1 &
    pids+=($!)
done

python convert/convert.py --model_dir $HF_MODEL_DIR --output_dir $OUTPUT_DIR --act_scales $ACT_SCALES --layer 61 --parts 9 --rank 8 > convert/convert_logs/log_8 2>&1
pids+=($!)

for pid in ${pids[@]}; do
    wait $pid
done

echo "All processes completed!"