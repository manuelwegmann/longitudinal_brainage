#!/bin/env bash

#SBATCH -A NAISS2025-22-353     # project ID found via "projinfo"
#SBATCH -p alvis                # what partition to use (usually not necessary)
#SBATCH -t 40:00:00          # how long time it will take to run
#SBATCH --gpus-per-node=V100:1    # choosing no. GPUs and their type

# load modules
module load virtualenv/20.26.2-GCCcore-13.3.0
module load matplotlib/3.9.2-gfbf-2024a
module load SciPy-bundle/2024.05-gfbf-2024a

source /mimer/NOBACKUP/groups/brainage/thesis_brainage/my_venv/bin/activate

# execute 
cd /mimer/NOBACKUP/groups/brainage/thesis_brainage/scripts
python -u run_CS_diagnosticgroups.py --run_name CS_diagnostic_groups \
    --max_epoch 30 \
    --epoch_weight_decay 10 \
    --dropout 0 \
    --lr 0.0001