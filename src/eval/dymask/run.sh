#!/bin/bash
#PBS -N cut3r_infer
#PBS -l select=1:ncpus=24:ngpus=4:mem=128gb:host=cvml07

# Activate the Conda environment
source /apps/miniconda3/etc/profile.d/conda.sh
# source /mnt/data/apps/miniconda3/etc/profile.d/conda.sh
conda activate cut3r

cd /home/ramanathan/Methods/CUT3R/src


# set -e

workdir='/mnt/rdata4_3/dymask_datasets/CUT3R'
model_name='segmentation_no_logits_fixed_mask+Spring' #'sparse_segmentation_no_logits_fixed_mask'
ckpt_name='checkpoint-best'
model_weights="${workdir}/checkpoints/${model_name}/dpt_512_vary_4_64_PO/${ckpt_name}.pth"
datasets=('davis') # 'PointOdyssey' 'Davis-16'

# for data in "${datasets[@]}"; do
output_dir="${workdir}/eval_results_new/dymask/davis_${model_name}_TTT3R"
echo "$output_dir"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 8 eval/dymask/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --eval_dataset 'davis' \
    --size 512
# done
