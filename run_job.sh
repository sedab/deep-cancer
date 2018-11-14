#!/bin/bash
#
#SBATCH --job-name=lusc
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --mem=100GB
#SBATCH --output=outputs/rq_train_%A.out
#SBATCH --error=outputs/rq_train_%A.err

module purge
module load python3/intel/3.5.3 pytorch/python3.5/0.2.0_3 torchvision/python3.5/0.1.9

cd /scratch/sb3923/deep-cancer/

python3 -u train.py $1 --experiment $2 > logs/$2.log
