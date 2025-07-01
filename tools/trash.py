import pandas as pd
import numpy as np
import os 

df = pd.read_csv('/mimer/NOBACKUP/groups/brainage/data/oasis3/participants.tsv', sep = '\t')
df = df[df['mr_sessions'] > 1]
sex = df['sex']

# Get unique values and counts from the 'sex' column
sex_counts = sex.value_counts()

print(sex_counts)

df = df[df['sex'] == 'U']
ids = df['participant_id']
valid_ids = []
i = len(ids)
j=0
for id in ids:
    path = os.path.join('/mimer/NOBACKUP/groups/brainage/data/oasis3', str(id), 'sessions.tsv')
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue  # Skip to next ID
    sessions_file = pd.read_csv(path, sep='\t')

    age_values = sessions_file['age'].dropna()

    if age_values.empty:
        print(f"No available (non-null) age values in {path}")
    else:
        print(f"{len(age_values)} age values found in {path}")
        j += 1
        valid_ids.append(id)

print(f'Of the {i} participants with sex U, {j} have available age information.')
