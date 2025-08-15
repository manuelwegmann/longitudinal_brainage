"""
Script to train and fit baseline models for the longitudinal brain age task.
"""

import pandas as pd
import numpy as np
import os

from loader import loader3D
import json
from argparse import Namespace

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr


def load_args_from_json(filepath):

    with open(filepath, 'r') as f:
        args_dict = json.load(f)

    args = Namespace(**args_dict)

    return args

opt = load_args_from_json('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC/run_details.json')

"""
Average prediction model
"""
output_dir = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/avg_model'
os.makedirs(output_dir, exist_ok=True)
blocks=[]

for i in range(5):
    train_fold = pd.read_csv(os.path.join(opt.folds_dir, f'train_fold_{i}.csv'))
    val_fold = pd.read_csv(os.path.join(opt.folds_dir, f'val_fold_{i}.csv'))
    train_fold = train_fold[['participant_id', 'sex']]
    val_fold = val_fold[['participant_id', 'sex']]
    train_data = loader3D(opt,train_fold).demo
    val_data = loader3D(opt,val_fold).demo

    mean_duration = train_data['duration'].mean()

    # Predict mean duration for all validation participants
    val_predictions = np.full(len(val_data), mean_duration)
    fold_indicator = np.full(len(val_data), f"fold_{i}")

    # Build results DataFrame
    block = pd.DataFrame({
        "Participant_ID": val_data['participant_id'],
        "Target": val_data['duration'],
        "Prediction": val_predictions,
        "Age": val_data['age'],
        "Sex (M)": val_data['sex_M'],
        "Session 1": val_data['session_id1'],
        "Session 2": val_data['session_id1'],
        "fold": fold_indicator
    })

    blocks.append(block)


combined_df = pd.concat(blocks, ignore_index=True)
combined_df.to_csv(os.path.join(output_dir, 'predictions_all_folds.csv'))

targets = combined_df['Target']
predictions = combined_df['Prediction']

mse = mean_squared_error(targets, predictions)
mae = mean_absolute_error(targets, predictions)

SSR = np.sum((targets - predictions) ** 2)
complete_avg = np.mean(targets)
complete_SSR = np.sum((targets - complete_avg) ** 2)

metrics = pd.DataFrame({
    'val_loss': [mse],
    'val_mae': [mae],
    'R2': [1 - SSR / complete_SSR]
})

# Define directory and file path
csv_path = os.path.join(output_dir, 'metrics.csv')
metrics.to_csv(csv_path, index=False)


"""
Regression model
"""
output_dir = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/reg_model'
os.makedirs(output_dir, exist_ok=True)
blocks = []

for i in range(5):
    train_fold = pd.read_csv(os.path.join(opt.folds_dir, f'train_fold_{i}.csv'))
    val_fold = pd.read_csv(os.path.join(opt.folds_dir, f'val_fold_{i}.csv'))
    train_fold = train_fold[['participant_id', 'sex']]
    val_fold = val_fold[['participant_id', 'sex']]

    # Load full data for train/val
    train_data = loader3D(opt, train_fold).demo
    val_data = loader3D(opt, val_fold).demo

    # Prepare regression inputs
    X_train = train_data[['age']].values  # shape (N, 1)
    y_train = train_data['duration'].values
    X_val = val_data[['age']].values
    y_val = val_data['duration'].values

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict duration on validation set
    val_predictions = model.predict(X_val)
    fold_indicator = np.full(len(val_data), f"fold_{i}")

    # Collect results
    block = pd.DataFrame({
        "Participant_ID": val_data['participant_id'],
        "Target": y_val,
        "Prediction": val_predictions,
        "Age": val_data['age'],
        "Sex (M)": val_data['sex_M'],
        "Session 1": val_data['session_id1'],
        "Session 2": val_data['session_id1'],  # Adjust if needed
        "fold": fold_indicator
    })

    blocks.append(block)

combined_df = pd.concat(blocks, ignore_index=True)
combined_df.to_csv(os.path.join(output_dir, 'predictions_all_folds.csv'))

# Metrics
targets = combined_df['Target']
predictions = combined_df['Prediction']

mse = mean_squared_error(targets, predictions)
mae = mean_absolute_error(targets, predictions)

SSR = np.sum((targets - predictions) ** 2)
complete_avg = np.mean(targets)
complete_SSR = np.sum((targets - complete_avg) ** 2)

metrics = pd.DataFrame({
    'val_loss': [mse],
    'val_mae': [mae],
    'R2': [1 - SSR / complete_SSR]
})

# Define directory and file path
csv_path = os.path.join(output_dir, 'metrics.csv')
metrics.to_csv(csv_path, index=False)
