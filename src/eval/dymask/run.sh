#!/bin/bash
#PBS -N cut3r_dynamic_training_no_logits_eval_fixed_mask
#PBS -l select=1:ncpus=16:ngpus=4:mem=128gb:host=cvml05

# Activate the Conda environment
source /mnt/data/apps/miniconda3/etc/profile.d/conda.sh
conda activate cut3r

cd /home/ramanathan/Methods/CUT3R/src


set -e

workdir='.'
model_name='segmentation_no_logits_fixed_mask'
ckpt_name='checkpoint-best'
model_weights="${workdir}/checkpoints/${model_name}/dpt_512_vary_4_64_PO/${ckpt_name}.pth"
datasets=('PointOdyssey' 'Davis-16' 'Davis-17' 'Davis-All')

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/dymask/${data}_${model_name}"
    echo "$output_dir"
    CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes 4 eval/dymask/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512
    python eval/dymask/eval_dymask.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --border_th 0.008
done
