#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=10GB
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

echo "creating filter visualizations for different layers of the architecture"

cd /scratch/sb3923/deep-cancer

python3 -u test_filter_viz.py 
