#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --gres=gpu:p100:1
#SBATCH --time=168:00:00
#SBATCH --mem=100GB
#SBATCH --error=outputs/rq_test_%A.err
#SBATCH --output=outputs/rq_test_%A.out


module purge
module load python3/intel/3.5.3
module load pytorch/python3.5/0.2.0_3
module load torchvision/python3.5/0.1.9

echo "Starting at `date`"
echo "Job name: $SLURM_JOB_NAME JobID: $SLURM_JOB_ID"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."

echo "experiment:"
echo $1
echo $2
echo $3

cd /scratch/sb3923/deep-cancer

python3 -u valid.py --data $1 --experiment $2 --model $3 > logs/$2_$3_valid.log
