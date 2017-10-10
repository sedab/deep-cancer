#!/bin/bash
#
#SBATCH --job-name=jmw_cap
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=47:00:00
#SBATCH --mem=20GB
#SBATCH --output=outputs/%A.out
#SBATCH --error=outputs/%A.err
#SBATCH --mail-user=jmw784@nyu.edu

module purge
module load python/intel/2.7.12
module load pillow/intel/4.0.0

pip install http://download.pytorch.org/whl/cu80/torch-0.1.11.post5-cp27-none-linux_x86_64.whl --user --upgrade
pip install torchvision --user --upgrade

cd /scratch/jmw784/capstone/Charrrrtreuse/

python -u model.py $1 > logs/$2.log
