#!/bin/bash
#SBATCH --nodes=5
#SBATCH --job-name=regressor
##SBATCH --partition=gpu-long
#SBATCH --mem=0
#SBATCH --nodelist=gn[04-08]
##SBATCH --exclusive
##SBATCH --gres=gpu:4

source ~/.bashrc
conda activate plifs


python3 -u rf_regressor.py

