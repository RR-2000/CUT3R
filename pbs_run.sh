#!/bin/bash
#PBS -N complex_cam_head
#PBS -l select=1:ncpus=16:ngpus=1:mem=64gb:host=cvml04

# Activate the Conda environment
# source /apps/miniconda3/etc/profile.d/conda.sh
source /mnt/data/apps/miniconda3/etc/profile.d/conda.sh
conda activate cut3r

cd /home/ramanathan/Methods/CUT3R/src

# Define tag for log and save path

#CUDA_LAUNCH_BLOCKING=1 NCCL_DEBUG=TRACE 
TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py  --config-name complex_cam_head


# # alias pn='pbsnodes -aSj'

# # pbsnodes -aSj

# # cvml01 NVIDIA RTX A5000

# # cvml03 NVIDIA RTX A5000

# # cvml06 NVIDIA RTX A5000

# # cvml05 Quadro RTX 8000

# # cvml07 Quadro RTX 8000

# # cvml08 NVIDIA NVIDIA GeForce RTX 2080 Ti

# # cvml10 NVIDIA RTX A5000

# # cvml11 NVIDIA A40

# # cvml12 NVIDIA RTX A5000

# cat check_gpu.sh

# #!/bin/bash

# #PBS -l select=1:host=cvml04:ngpus=1:mem=1gb

 

# echo "Running on node: $(hostname)"

# nvidia-smi

# #!/bin/bash

# #PBS -N dynamic_mask_training

# #PBS -l select=1:ncpus=26:ngpus=4:mem=180gb:host=cvml03

 

# # Activate the Conda environment

# source /mnt/data/apps/miniconda3/etc/profile.d/conda.sh

# conda activate fast3r

 

# cd /mnt/rdata4_6/kx_data/fast3r_dymask/code

 

# # Define tag for log and save path

# TAG="dynamic_hint_vos_20250603_fix_vos"

 
# OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python demo.py --input demo_data/dog-gooses --output_dir demo_tmp --seq_name dog-gooses --weights checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth 

# ps -eo pid,user,pcpu,pmem,comm --sort=-pcpu | grep ramanat