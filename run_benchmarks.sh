#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --partition=p_general
#SBATCH --qos=high
#SBATCH -o slurm_out

conda env update -f ./environment.yml
source activate pose_estimation

pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html

module load cuda/10.1
module load cudnn/7.6.5_for_cuda_10.1

cd src/
srun python benchmark.py -f /shared/sense/ana/videos_anotados

