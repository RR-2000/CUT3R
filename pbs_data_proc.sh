#!/bin/bash
#PBS -N rdata4_5_HOI4D_clearing
#PBS -l select=1:ncpus=32:ngpus=0:mem=64gb:host=cvml02

# Activate the Conda environment
source /apps/miniconda3/etc/profile.d/conda.sh
# source /mnt/data/apps/miniconda3/etc/profile.d/conda.sh
conda activate cut3r

cd /home/ramanathan/Methods/CUT3R/src

export PYTHONPATH=/home/ramanathan/Methods/CUT3R:$PYTHONPATH

rm -rf /mnt/rdata4_5/HOI4D

# python /home/ramanathan/Methods/CUT3R/datasets_preprocess/preprocess_hoi4d.py --root_dir /mnt/rdata4_3/dymask_datasets/HOI4D/HOI4D_release --cam_root /mnt/rdata4_3/dymask_datasets/HOI4D/camera_params --out_dir /mnt/rdata4_5/HOI4D --max_workers 20

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