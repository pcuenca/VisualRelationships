#!/bin/sh

# dataset from 'nlvr2', 'adobe', 'spotdiff'
dataset=inference

# Needs to match the type of model we trained
model=dynamic
model_snapshot=snap/adobe/speaker/dynamic_2pixel/best_eval
task=speaker_inference
if [ -z "$1" ]; then
    gpu=0
else
    gpu=$1
fi

log_dir=$dataset/$task/$name
mkdir -p snap/$log_dir
mkdir -p log/$dataset/$task
cp $0 snap/$log_dir/run.bash
cp -r src snap/$log_dir/src

CUDA_VISIBLE_DEVICES=$gpu stdbuf -i0 -o0 -e0 python src/inference.py --output snap/$log_dir \
    --maxInput 40 --model $model --worker 16 --train speaker_inference --dataset $dataset \
    --batchSize 95 --hidDim 512 --dropout 0.5 \
    --seed 9595 \
	--load ${model_snapshot} \
    | tee log/$log_dir.log

