#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b"
OCRBENCH_DIR="./playground/data/eval/ocrbench"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_ocrbench infer \
        --model-path liuhaotian/llava-v1.5-7b \
        --question-file ${OCRBENCH_DIR}/OCRBench.json \
        --image-folder ${OCRBENCH_DIR}/data \
        --answers-file ${OCRBENCH_DIR}/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        "$@" &
done

wait

output_file=${OCRBENCH_DIR}/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${OCRBENCH_DIR}/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python -m llava.eval.model_vqa_ocrbench eval \
    --results-file $output_file
