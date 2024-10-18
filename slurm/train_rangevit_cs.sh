#!/bin/bash

#SBATCH -n 16
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --tmp=100G
#SBATCH --gpus=4
#SBATCH --gres=gpumem:11G
#SBATCH --job-name=rangevit-cityscapes-nuscenes
#SBATCH --output=/cluster/work/rsl/patelm/result/slurm_output/%x_%j.out
#SBATCH --error=/cluster/work/rsl/patelm/result/slurm_output/%x_%j.err

source ~/.bashrc
conda activate dinov2
echo "Preparing the dataset"

module load eth_proxy
rangevit_dir=/cluster/home/patelm/ws/rsl/rangevit
experiment_cfg=config_nusc_cs
num_gpus=4
output_path=/cluster/work/rsl/patelm/result/rangevit
experiment_name=NuScenes_CS_ViT-S

cd $rangevit_dir

# Prepare the dataset (This is for full original dataset which we dont need)
# ./slurm/organize_data.sh /cluster/scratch/patelm/Nuscenes_full ${TMPDIR}/Nuscenes
tar -xvf /cluster/scratch/patelm/Nuscenes.tar -C ${TMPDIR}

echo "Cuda visible devices ${CUDA_VISIBLE_DEVICES}"

python -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port=63545 --use_env main.py "${experiment_cfg}.yaml" --data_root "${TMPDIR}/Nuscenes_data" --save_path "${output_path}" --pretrained_model "${rangevit_dir}/pretrained_models/segmenter_checkpoint.pth" --id "${experiment_name}"

# python -m torch.distributed.launch --nproc_per_node=2 --master_port=63545 --use_env main.py "config_nusc.yaml" --data_root "${TMPDIR}/Nuscenes" --save_path "/cluster/work/rsl/patelm/result/rangevit" --pretrained_model "/cluster/home/patelm/ws/rsl/rangevit/pretrained_models/dinov2_vits14_pretrain.pth" --id "NuScenes_Dinov2_ViT-S"
