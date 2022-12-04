#!/bin/bash

#SBATCH --partition=normal
#SBATCH --job-name=plifCN_test
#SBATCH --ntasks=50
##SBATCH --mem=0

source ~/.bashrc
conda activate plifs
export LD_LIBRARY_PATH=/home/hmslati/.conda/envs/plifs/lib/:$LD_LIBRARY_PATH

ln -sf /home/hmslati/.conda/envs/plifs/lib/libstdc++.so.6 /home/hmslati/.conda/envs/plifs/lib/python3.8/site-packages/openbabel/
python plif_cnn_test_2.py $1 $2
