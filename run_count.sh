#!/bin/bash
#
#SBATCH --job-name=counting
#SBATCH --gres=gpu:1
#SBATCH --time=47:00:00
#SBATCH --mem=15GB
#SBATCH --output=outputs/counting_%A.out
#SBATCH --error=outputs/counting_%A.err

module purge
module load python3/intel/3.5.3
module load pytorch/python3.5/0.2.0_3
module load torchvision/python3.5/0.1.9

cd /scratch/sb3923/deep-cancer

python3 -u count_files.py --data $1 > logs/$1_count.log
