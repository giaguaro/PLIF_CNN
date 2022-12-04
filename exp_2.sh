#!/bin/bash
#SBATCH --job-name=LrgSklCNN
#SBATCH --nodes=3
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --nodelist=gn[13,19-20]
##SBATCH --nodelist=gn[04-11,13-23,25-26]
#SBATCH --mem=0

export WORLD_SIZE=12
export MASTER_PORT=12346
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed
source ~/.bashrc
conda activate plifs

ip1=`hostname -I | awk '{print $1}'`
echo $ip1
echo "tcp://${ip1}:${MASTER_PORT}"

### the command to run
#srun python -u main_distributed.py --world_size 12 --dist-backend nccl --multiprocessing-distributed --dist-file dist_file


CUDA_VISIBLE_DEVICES=0,1,2,3
python -u -m torch.distributed.launch \
    --nnodes=3 \
    --master_addr=$ip1 \
    --nproc_per_node=4 \
    --master_port=2334 \
    main_distributed.py --multiprocessing-distributed --dist-file dist_file --dist-backend nccl 
