#!/bin/bash
#
#SBATCH --job-name=tr_all_ds
#SBATCH --gres=gpu:p100:1 
#SBATCH --time=168:00:00
#SBATCH --mem=100GB
#SBATCH --output=outputs/rq_train_all_ds%A.out
#SBATCH --error=outputs/rq_train_all_ds%A.err

module purge
module load python3/intel/3.5.3 pytorch/python3.5/0.2.0_3 torchvision/python3.5/0.1.9

cd /scratch/sb3923/deep-cancer/

python3 -u train_all.py $1 --experiment $2 > logs/$2.log
