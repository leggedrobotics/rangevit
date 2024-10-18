#!/bin/bash

#SBATCH -n 12
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --tmp=500G
#SBATCH --gpus=4
#SBATCH --gres=gpumem:38G
#SBATCH --job-name=imagenet-dino
#SBATCH --output=/cluster/work/rsl/patelm/result/slurm_output/%x_%j.out
#SBATCH --error=/cluster/work/rsl/patelm/result/slurm_output/%x_%j.err

source ~/.bashrc
conda activate dinov2
echo "Preparing the dataset"

cd /cluster/home/patelm/ws/rsl/rangevit

# Prepare the dataset
./slurm/organize_data.sh /cluster/scratch/patelm/Nuscenes ${TMPDIR}/Nuscenes

echo "Cuda visible devices ${CUDA_VISIBLE_DEVICES}"

python -m torch.distributed.launch --nproc_per_node=2 --master_port=63545 --use_env main.py 'config_nusc.yaml' --data_root '/media/patelm/ssd/Nuscenes_data' --save_path '/media/patelm/ssd/rangevit_output' --pretrained_model '/home/patelm/Downloads/dinov2_vits14_pretrain.pth' --id 'NuScenes_Dinov2_ViT-S'
