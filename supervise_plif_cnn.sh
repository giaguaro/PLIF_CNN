#!/bin/bash

#SBATCH --partition=normal
#SBATCH --job-name=plifs_cnn
#SBATCH --ntasks=1
##SBATCH --mem=0

source ~/.bashrc
conda activate plifs
export LD_LIBRARY_PATH=/home/hmslati/.conda/envs/plifs/lib/:$LD_LIBRARY_PATH

ln -sf /home/hmslati/.conda/envs/plifs/lib/libstdc++.so.6 /home/hmslati/.conda/envs/plifs/lib/python3.8/site-packages/openbabel/


while read -r line; 
do 
	while [[ $(squeue -u hmslati --name plifCN_train | wc -l) -gt 200 ]]; do echo "waiting for freed up cpus to finish"; sleep 1s; done;
	v1=$(echo $line | cut -d"," -f1); 
	v2=$(echo $line | cut -d"," -f5); 
	echo "$v1 $v2";
	if [ ! -f ./train_2/${v1}*pkl ]; then sbatch supervise_plif_cnn_train.sh $v1 $v2; else :;fi
done <train_df_slurm_unprocessed.csv

while read -r line;
do
        while [[ $(squeue -u hmslati --name plifCN_test | wc -l) -gt 200 ]]; do echo "waiting for freed up cpus to finish"; sleep 1s; done;
        v1=$(echo $line | cut -d"," -f1);
        v2=$(echo $line | cut -d"," -f5);
        echo "$v1 $v2";
        if [ ! -f ./test/${v1}*pkl ]; then sbatch supervise_plif_cnn_test.sh $v1 $v2; else :;fi
done <test_df_slurm.csv
