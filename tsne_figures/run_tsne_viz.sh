#!/bin/bash
#
#SBATCH --job-name=tsne
#SBATCH --gres=gpu:1
#SBATCH --time=46:00:00
#SBATCH --mem=100GB
#SBATCH --output=outputs/tsne_%A.out
#SBATCH --error=outputs/tsne_%A.err

module purge
module load python3/intel/3.5.3
module load pytorch/python3.5/0.2.0_3
module load torchvision/python3.5/0.1.9

cd /scratch/jmw784/capstone/deep-cancer/tsne_figures

python3 -u tsne_viz.py  > logs/tsne_$1.log
