#!/bin/bash
#SBATCH --job-name=LrgSklCNN

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=20
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
##SBATCH --nodelist=gn[04-05]
##SBATCH --nodelist=gn[25-26,02,04-11,13-23]
#SBATCH --nodelist=gn[05-11,13-23,25-26]
#SBATCH --mem=0
#SBATCH --chdir=/groups/cherkasvgrp/share/progressive_docking/hmslati/plif_cnn/
#SBATCH --output=/groups/cherkasvgrp/share/progressive_docking/hmslati/plif_cnn/%x-%j.out

module purge
module load pytorch-gpu/py3/1.5.0

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=80

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed
source ~/.bashrc
conda activate plifs

### the command to run
srun python -u main_final.py --net cnn --lr-scheduler --resume ./checkpoint.pth.tar
