#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --gres=gpu:p40:1
#SBATCH --time=168:00:00
#SBATCH --mem=100GB
#SBATCH --error=outputs/rq_test_%A.err

module purge
module load python3/intel/3.5.3
module load pytorch/python3.5/0.2.0_3
module load torchvision/python3.5/0.1.9

echo "Starting at `date`"
echo "Job name: $SLURM_JOB_NAME JobID: $SLURM_JOB_ID"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."

cd /scratch/sb3923/deep-cancer

python3 -u test.py --data $1 --experiment $2 --model $3 > logs/$2_$3_test.log
