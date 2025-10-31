import sys
import pandas as pd
import numpy as np
import os

save_dir = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/CI_CN_groups'

control_group = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/CI_CN_groups/CN_participants.csv')

all_participants1 = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/cs_train_fold_0.csv')
all_participants2 = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/cs_val_fold_0.csv')
participants_cn = pd.concat([all_participants1, all_participants2], ignore_index=True)
print(len(participants_cn))

ids_control = control_group['participant_id'].values

filtered_clean_participants = participants_cn[~participants_cn['participant_id'].isin(ids_control)]
filtered_clean_participants.to_csv(os.path.join(save_dir, 'CS_CN_training_for_CI.csv'), index=False)