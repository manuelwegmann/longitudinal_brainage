"""
Script to concatenate predictions from multiple folds into a single DataFrame for evaluation
"""

import os
import pandas as pd

path_to_folds = 'blank'  # Replace with  actual path to folds dir

# List to store individual DataFrames
dfs = []

# Iterate through each item in the directory
for folder_name in os.listdir(path_to_folds):
    if folder_name.startswith("fold_"):
        folder_path = os.path.join(path_to_folds, folder_name)
        csv_path = os.path.join(folder_path, "predictions_val.csv")
        
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            df['fold'] = folder_name
            dfs.append(df)

# Combine all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)

# Preview the result
combined_df.to_csv(os.path.join(path_to_folds, 'predictions_all_folds.csv'))
