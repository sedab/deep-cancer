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
module load pytorch/python3.5/0.2.0_3
module load torchvision/python3.5/0.1.9


cd /scratch/rds491/capstone/Charrrrtreuse/

python3 -u train_inception.py $1 --experiment $2 > logs/$2.log


