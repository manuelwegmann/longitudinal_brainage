import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

results_path = "/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_plus_age/results_all_folds.csv"
df = pd.read_csv(results_path)
outlier_threshold = 5
mae = np.abs(df["Target"].values - df["Prediction"].values)
df["MAE"] = mae
print(f"Mean absolute error with sessions: {df['MAE'].mean()}")
outliers = df[df["MAE"] > outlier_threshold]
print(outliers.head())
print(f"Number of outliers: {len(outliers)}")
print("We will now see if the outliers have something to do with field strength.")

weak_df = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/further_analysis_results/weak_participants.csv')

not_weak_count = 0

for index, row in outliers.iterrows():
    if row['Participant_ID'] not in weak_df['weak_participants'].values:
        not_weak_count += 1

print(f"Number of outliers that are not weak participants: {not_weak_count}")

weak_count = 0

for index, row in outliers.iterrows():
    if row['Participant_ID'] in weak_df['weak_participants'].values:
        relevant_case = weak_df[weak_df['weak_participants'] == row['Participant_ID']]
        if row['Session 1'] in relevant_case['weak_ids'].values or row['Session 2'] in relevant_case['weak_ids'].values:
            weak_count += 1

print(f"Number of outliers that are weak participants: {weak_count}")

valid_rows = []
for index, row in df.iterrows():
    if row['Participant_ID'] in df['Participant_ID'].values:
        relevant_case = weak_df[weak_df['weak_participants'] == row['Participant_ID']]
        if row['Session 1'] in relevant_case['weak_ids'].values or row['Session 2'] in relevant_case['weak_ids'].values:
            continue
    valid_rows.append(row)
            
df_valid = pd.DataFrame(valid_rows)
print(f"Mean absolute error without weak sessions: {df_valid['MAE'].mean()}")
