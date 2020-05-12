#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --partition=p_general
#SBATCH --qos=high
#SBATCH -o slurm_out

conda env create -f environment.yml
source activate pose_estimation

echo $CUDA_VISIBLE_DEVICES

module load cuda/10.2
module load cudnn/7.6.5_for_cuda_10.2

cd src/
srun python benchmark.py -f /shared/sense/ana/videos_anotados

