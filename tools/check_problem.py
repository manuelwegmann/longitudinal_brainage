import numpy as np
import pandas as pd
import os

file = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/participants.csv')
print(file.head())
ids = file['participant_id'].to_list()
count = 0
for id in ids:
    path = f'/mimer/NOBACKUP/groups/brainage/data/oasis3/{id}'
    sessions_file = pd.read_csv(os.path.join(path, 'sessions.tsv'), sep = '\t')
    if not sessions_file['cognitiveyly_normal'].all():
        print("Problem with id:", id)
        count += 1
if count == 0:
    print("No problems found")
else:
    print("Total problems found:", count)