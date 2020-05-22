#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --partition=p_general
#SBATCH --qos=high
#SBATCH -o slurm_out

conda remove --name pose_estimation --all
conda env create -f ./environment.yml
source activate pose_estimation

srun python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
srun python -m pip install 'git+https://github.com/MVIG-SJTU/AlphaPose.git'

module load cuda/10.1
module load cudnn/7.6.5_for_cuda_10.1

cd src/
srun python benchmark.py -f /shared/sense/ana/videos_anotados

