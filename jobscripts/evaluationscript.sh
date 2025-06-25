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
cd /mimer/NOBACKUP/groups/brainage/thesis_brainage/tools

python -u evaluation_pipeline.py --results_file /mimer/NOBACKUP/groups/brainage/thesis_brainage/results/fs_LILAC_plus/fold_0/results_val.csv
python -u evaluation_pipeline.py --results_file /mimer/NOBACKUP/groups/brainage/thesis_brainage/results/fs_LILAC_plus/fold_1/results_val.csv
python -u evaluation_pipeline.py --results_file /mimer/NOBACKUP/groups/brainage/thesis_brainage/results/fs_LILAC_plus/results_all_folds.csv