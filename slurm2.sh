#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 72:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:v100:1
#SBATCH -J fp
#SBATCH -o fp.out.%j
#SBATCH -e fp.err.%j
#SBATCH --account=project_2002605
#SBATCH

module purge
module load pytorch/1.9

python final_project.py rms_prop_optimizer
python final_project.py adam_optimizer
