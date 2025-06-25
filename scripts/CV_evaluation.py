import os
import pandas as pd

path_to_folds = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/fs_LILAC_plus'

# List to store individual DataFrames
dfs = []

# Iterate through each item in the directory
for folder_name in os.listdir(path_to_folds):
    if folder_name.startswith("fold_"):
        folder_path = os.path.join(path_to_folds, folder_name)
        csv_path = os.path.join(folder_path, "results_val.csv")
        
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            df['fold'] = folder_name  # Optionally tag the fold
            dfs.append(df)

# Combine all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)

# Preview the result
combined_df.to_csv(os.path.join(path_to_folds, 'results_all_folds.csv'))
