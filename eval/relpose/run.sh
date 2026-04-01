# #!/bin/bash
# #PBS -N Pose_Eval
# #PBS -l select=1:ncpus=16:ngpus=4:mem=256gb:host=cvml07

# # Activate the Conda environment
# source /apps/miniconda3/etc/profile.d/conda.sh
# # source /mnt/data/apps/miniconda3/etc/profile.d/conda.sh
# conda activate cut3r

# cd /home/ramanathan/Methods/CUT3R

# set -e

# workdir='.'
# model_name='Baseline'
# ckpt_name='cut3r_512_dpt_4_64'
# model_weights="${workdir}/src/${ckpt_name}.pth"
# datasets=('scannet')


# for data in "${datasets[@]}"; do
#     output_dir="${workdir}/eval_results/relpose/${data}_${model_name}"
#     echo "$output_dir"
#     accelerate launch --num_processes 4 --main_process_port 29558 eval/relpose/launch.py \
#         --weights "$model_weights" \
#         --output_dir "$output_dir" \
#         --eval_dataset "$data" \
#         --size 512
# done

#!/bin/bash
#PBS -N Pose_Eval
#PBS -l select=1:ncpus=16:ngpus=1:mem=128gb:host=cvml08

# Activate the Conda environment
source /apps/miniconda3/etc/profile.d/conda.sh
# source /mnt/data/apps/miniconda3/etc/profile.d/conda.sh
conda activate cut3r

cd /home/ramanathan/Methods/CUT3R

nvidia-smi

set -e
export PYTHONPATH=/home/ramanathan/Methods/CUT3R:/home/ramanathan/Methods/CUT3R/src:$PYTHONPATH

workdir='/mnt/rdata4_3/dymask_datasets/CUT3R/checkpoints'
evaldir='/mnt/rdata4_3/dymask_datasets/CUT3R/eval_prototype'
model_name='mask_attn_individual_shallow_Regularized'
ckpt_name='checkpoint-last'
model_weights="${workdir}/${model_name}/dpt_512_vary_4_64_PO/${ckpt_name}.pth"
datasets=('sintel') # 'scannet' 'tum' 'sintel' 


for data in "${datasets[@]}"; do
    output_dir="${evaldir}/testing_mlp_mask/${data}_${model_name}"
    echo "$output_dir"
    accelerate launch --num_processes 1 --main_process_port 29558 eval/relpose/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512
done



