import sys
import pandas as pd
import numpy as np
import os
import argparse
from argparse import Namespace
import json

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.append(scripts_path)
from loader import loader3D

#function to load model parameters
def load_args_from_json(filepath):

    with open(filepath, 'r') as f:
        args_dict = json.load(f)

    args = Namespace(**args_dict)

    return args

def find_closest_match(age, duration, sex, df, used_ids):
    error = np.inf
    closest_row = None
    df_gender = df[df['sex_M']==sex]
    for _, row in df_gender.iterrows():
        error_new = (age-row['age'])**2 + (duration-row['duration'])**2
        if error_new < error and row['participant_id'] not in used_ids:
            error = error_new
            closest_row = row.copy()
            closest_row['error'] = error_new
    return pd.DataFrame([closest_row])

args = load_args_from_json('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC/run_details.json')

participants_ci = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/CI_participants.csv')
all_participants1 = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/train_fold_0.csv')
all_participants2 = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/val_fold_0.csv')
participants_cn = pd.concat([all_participants1, all_participants2], ignore_index=True)

print("We are now going to load the CI participants scan pairs with filtered field strengths.")
df_CI = loader3D(args, participants_ci).demo
print("We are now going to load the CN participants scan pairs with filtered field strengths.")
df_CN = loader3D(args, participants_cn).demo

idx = df_CI.groupby('participant_id')['duration'].idxmax()
valid_CI = df_CI.loc[idx].reset_index(drop=True)

idx = df_CN.groupby('participant_id')['duration'].idxmax()
valid_CN = df_CN.loc[idx].reset_index(drop=True)

closest_rows = []
used_participants = []
for _, row in valid_CI.iterrows():
    closest_row = find_closest_match(row['age'], row['duration'], row['sex_M'], valid_CN, used_participants)
    closest_rows.append(closest_row)
    used_participants.append(closest_row['participant_id'].values[0])

closest_df = pd.concat(closest_rows, ignore_index=True)
print(f"Unique Ids in CI: {valid_CI['participant_id'].nunique()}")
print(f"Unique Ids in CN: {closest_df['participant_id'].nunique()}")

ids_control = closest_df['participant_id'].values
filtered_clean_participants = participants_cn[~participants_cn['participant_id'].isin(ids_control)]
print(f"Unique Ids in CN training for CI: {filtered_clean_participants['participant_id'].nunique()}")

valid_CI['sex'] = np.where(valid_CI['sex_M'] == 1, 'M', 'F')
closest_df['sex'] = np.where(closest_df['sex_M'] == 1, 'M', 'F')

save_dir = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/CI_CN_groups'
valid_CI.to_csv(os.path.join(save_dir, 'CI_participants.csv'), index=False)
closest_df.to_csv(os.path.join(save_dir, 'CN_participants.csv'), index=False)
filtered_clean_participants.to_csv(os.path.join(save_dir, 'CN_training_for_CI.csv'), index=False)
    


