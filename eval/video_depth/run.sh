#!/bin/bash
#PBS -N Video_Depth_Eval
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb:host=cvml08

# # Activate the Conda environment
source /apps/miniconda3/etc/profile.d/conda.sh
# # source /mnt/data/apps/miniconda3/etc/profile.d/conda.sh
conda activate cut3r

cd /home/ramanathan/TTT/CUT3R

nvidia-smi

set -e
export PYTHONPATH=/home/ramanathan/TTT/CUT3R:/home/ramanathan/TTT/CUT3R/src:$PYTHONPATH

workdir='/mnt/rdata4_3/dymask_datasets/CUT3R/checkpoints'
# workdir='.'
evaldir='/mnt/rdata4_3/dymask_datasets/CUT3R/eval_prototype'
model_name='baseline'
# model_name='base'
ckpt_name='cut3r_512_dpt_4_64'
model_weights="${workdir}/src/${ckpt_name}.pth"
# model_weights="${workdir}/${model_name}/dpt_512_vary_4_64_PO/${ckpt_name}.pth"
model_weights="/home/ramanathan/TTT/CUT3R/src/cut3r_512_dpt_4_64.pth"
datasets=('sintel' 'bonn' ) # 'sintel' 'bonn' 'kitti'

for data in "${datasets[@]}"; do
    output_dir="${evaldir}/testing_mlp_mask/video_depth/${data}_${model_name}"
    echo "$output_dir"
    accelerate launch --num_processes 1  eval/video_depth/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512 \
        --TTT3R 
    python eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --align "scale"
done
