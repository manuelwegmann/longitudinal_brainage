#!/bin/env bash

#SBATCH -A NAISS2025-22-353     # project ID found via "projinfo"
#SBATCH -p alvis                # what partition to use (usually not necessary)
#SBATCH -t 47:59:59          # how long time it will take to run
#SBATCH --gpus-per-node=A40:1    # choosing no. GPUs and their type

# load modules
module load virtualenv/20.26.2-GCCcore-13.3.0
module load matplotlib/3.9.2-gfbf-2024a
module load SciPy-bundle/2024.05-gfbf-2024a

source /mimer/NOBACKUP/groups/brainage/thesis_brainage/my_venv/bin/activate

# execute 
cd /mimer/NOBACKUP/groups/brainage/thesis_brainage/scripts
python -u run_LILAC.py --batchsize 16 --early_stopping_patience 10 --seed 20 --run_name '5-fold-cv_w_age_dropout' --folds 5 --dropout 0.2