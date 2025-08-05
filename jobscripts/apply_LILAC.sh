#!/bin/env bash

#SBATCH -A NAISS2025-22-353     # project ID found via "projinfo"
#SBATCH -p alvis                # what partition to use (usually not necessary)
#SBATCH -t 05:00:00          # how long time it will take to run
#SBATCH --gpus-per-node=V100:1    # choosing no. GPUs and their type

# load modules
module load virtualenv/20.26.2-GCCcore-13.3.0
module load matplotlib/3.9.2-gfbf-2024a
module load SciPy-bundle/2024.05-gfbf-2024a

source /mimer/NOBACKUP/groups/brainage/thesis_brainage/my_venv/bin/activate

# execute 
cd /mimer/NOBACKUP/groups/brainage/thesis_brainage/scripts

python -u apply_LILAC.py --json /mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_CI/run_details.json \
    --model LILAC \
    --participants_file /mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/CI_participants.csv \
    --model_state /mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_CI/model.pt \
    --CI yes

python -u apply_LILAC.py --json /mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_CI/run_details.json \
    --model LILAC \
    --participants_file /mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/CN_controlgroup.csv \
    --model_state /mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_CI/model.pt \
    --CI yes
