#!/bin/bash
#PBS -N cut3r_infer
#PBS -l select=1:ncpus=32:ngpus=4:mem=128gb:host=cvml07

# Activate the Conda environment
source /apps/miniconda3/etc/profile.d/conda.sh
# source /mnt/data/apps/miniconda3/etc/profile.d/conda.sh
conda activate cut3r

cd /home/ramanathan/Methods/CUT3R/src


# set -e

workdir='/mnt/rdata4_3/dymask_datasets/CUT3R'
model_name='no_kubric_prototype_full' #'sparse_segmentation_no_logits_fixed_mask'
ckpt_name='checkpoint-final'
model_weights="${workdir}/checkpoints/${model_name}/dpt_512_vary_4_64_PO/${ckpt_name}.pth"
datasets=('davis' 'segTrackv2') # 'kubric-custom' 'PointOdyssey' 'davis' 'segTrackv2' 'FBMS-56'

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_prototype/dymask/${data}_${model_name}"
    echo "$output_dir"
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 8 eval/dymask/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512 \
        # --TTT3R
done
