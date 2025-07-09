import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from loader import loader3D
import json
from argparse import Namespace


def load_args_from_json(filepath):

    with open(filepath, 'r') as f:
        args_dict = json.load(f)

    args = Namespace(**args_dict)

    return args

args = load_args_from_json('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_age/run_details.json')
args.project_data_dir = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/data'
args.optional_meta = ['sex_M']

participant_df = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/further_analysis_results/participants_file.csv')
train_df, test_df = train_test_split(participant_df, test_size=0.2, random_state=42)
train_data = loader3D(args,train_df).demo
test_data = loader3D(args,test_df).demo

# ----- Naive Mean Baseline -----
naive_pred = np.mean(train_data['duration'])

loss_train = np.mean((train_data['duration'] - naive_pred) ** 2)
loss_test = np.mean((test_data['duration'] - naive_pred) ** 2)

mae_train = np.mean(np.abs(train_data['duration'] - naive_pred))
mae_test = np.mean(np.abs(test_data['duration'] - naive_pred))

print("Naive Mean Baseline:")
print(f"Train MSE: {loss_train:.3f}, MAE: {mae_train:.3f}")
print(f"Test MSE: {loss_test:.3f}, MAE: {mae_test:.3f}")

# ----- Naive Linear Regression using Age -----

# Fit simple linear regression: duration = a * age + b
X_train = train_data['age'].values
y_train = train_data['duration'].values

# Fit using least squares
A = np.vstack([X_train, np.ones(len(X_train))]).T  # Design matrix
a, b = np.linalg.lstsq(A, y_train, rcond=None)[0]   # Solve for coefficients

# Predict on train and test
y_train_pred = a * train_data['age'] + b
y_test_pred = a * test_data['age'] + b

# Evaluate
mse_train_age = np.mean((train_data['duration'] - y_train_pred) ** 2)
mse_test_age = np.mean((test_data['duration'] - y_test_pred) ** 2)

mae_train_age = np.mean(np.abs(train_data['duration'] - y_train_pred))
mae_test_age = np.mean(np.abs(test_data['duration'] - y_test_pred))

print("\nLinear Regression using Age:")
print(f"Train MSE: {mse_train_age:.3f}, MAE: {mae_train_age:.3f}")
print(f"Test MSE: {mse_test_age:.3f}, MAE: {mae_test_age:.3f}")

