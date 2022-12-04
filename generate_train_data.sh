#!/bin/bash

#SBATCH --partition=normal
#SBATCH --job-name=plifs_cnn
#SBATCH --ntasks=5
##SBATCH --mem=0

source ~/.bashrc
conda activate plifs

python -u generate_train_data.py


