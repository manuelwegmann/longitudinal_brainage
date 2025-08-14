import pandas as pd
import os
import numpy as np

p_CI = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/CI_participants.csv')
r_CI = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_CI/predictions_CI.csv')
p_CN = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/CN_controlgroup.csv')
r_CN = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_CI/predictions_CN.csv')

# CI group
missing_in_results_CI = set(p_CI['participant_id']) - set(r_CI['Participant_ID'])
missing_in_participants_CI = set(r_CI['Participant_ID']) - set(p_CI['participant_id'])

# CN group
missing_in_results_CN = set(p_CN['participant_id']) - set(r_CN['Participant_ID'])
missing_in_participants_CN = set(r_CN['Participant_ID']) - set(p_CN['participant_id'])

print("CI participants missing from results:", missing_in_results_CI)
print("Results with CI IDs not in participants:", missing_in_participants_CI)

print("CN participants missing from results:", missing_in_results_CN)
print("Results with CN IDs not in participants:", missing_in_participants_CN)

print(len(r_CI))

def find_closest_match(age, duration, sex, df, used_ids):
    error = np.inf
    closest_row = None
    df_gender = df[df['Sex (M)'] == sex]
    for _, row in df_gender.iterrows():
        error_new = (age - row['Age'])**2 + (duration - row['Target'])**2
        if error_new < error and row['Participant_ID'] not in used_ids:
            error = error_new
            closest_row = row.copy()
    return pd.DataFrame([closest_row]) if closest_row is not None else None, error

used_ids = []
total_error = 0
rows_to_delete = []

for idx, row in r_CI.iterrows():
    age = row['Age']
    duration = row['Target']
    sex = row['Sex (M)']
    closest_row, error = find_closest_match(age, duration, sex, r_CN, used_ids)

    if closest_row is not None and not closest_row.empty:
        total_error += error
        used_ids.append(closest_row['Participant_ID'].values[0])
    else:
        rows_to_delete.append(idx)  # Mark for deletion

# Remove the rows after the loop
r_CI = r_CI.drop(rows_to_delete).reset_index(drop=True)

print(total_error / len(r_CI))
print(len(r_CI))