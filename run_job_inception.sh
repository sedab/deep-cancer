#!/bin/bash
#
#SBATCH --job-name=charrrr_inception
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem=40GB
#SBATCH --output=outputs/%A.out
#SBATCH --error=outputs/%A.err

module purge
module load python3/intel/3.5.3
python3 -m pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl --user --upgrade
python3 -m pip install torchvision --user --upgrade
python3 -m pip install comet_ml --user

cd /scratch/rds491/capstone/Charrrrtreuse/

python3 -u inception_transfer_learning.py > logs/inception_debug.log
