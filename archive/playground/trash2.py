import pandas as pd
import numpy as np
import os

results = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/CI_CN_model_comparison.csv')
file = pd.read_csv('/mimer/NOBACKUP/groups/brainage/data/oasis3/participants.tsv', sep='\t')
file = file.rename(columns={'participant_id': 'Participant_ID'})
merged = results.merge(file[['Participant_ID', 'sex']], on='Participant_ID', how='left')

CN = merged[merged['Group'] == 'CN']
CI = merged[merged['Group'] == 'CI']

print("CN sex counts:")
print(CN['sex'].value_counts())

print("\nCI sex counts:")
print(CI['sex'].value_counts())

merged.to_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/CI_CN_model_comparison.csv', index=False)