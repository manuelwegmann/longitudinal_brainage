from loader_AE import loader3D
import numpy as np
import pandas as pd
import os
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_directory', default='/mimer/NOBACKUP/groups/brainage/data/oasis3', type=str, help="directory of the data (OASIS3)")
    parser.add_argument('--project_data_dir', default ='/mimer/NOBACKUP/groups/brainage/thesis_brainage/data', type=str, help="directory with the updated session files")
    parser.add_argument('--participants_file_path', default = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/further_analysis_results/participants_file_with_age.csv', type = str, help = 'path to participants csv.')

    parser.add_argument('--model', default='LILAC_plus', type=str, choices=['LILAC', 'LILAC_plus'], help="model to use: LILAC or LILAC_plus")

    #data preprocessing arguments
    parser.add_argument('--compression', default=8, type=int, help='compression used in autoencoder (4 or 8)')
    parser.add_argument('--image_channel', default=1, type=int, help="number of channels in the input image")
    parser.add_argument('--seed', default=15, type=int)

    #target and optional meta data arguments
    parser.add_argument('--target_name', default='duration', type=str, help="name of the target variable")
    parser.add_argument('--optional_meta', nargs='+', default=['sex_M'], help="List of optional meta to be used in the model")
    
    #model architecture arguments
    parser.add_argument('--n_of_blocks', default=4, type=int, help="number of blocks in the encoder")
    parser.add_argument('--initial_channel', default=16, type=int, help="initial channel size after first conv")
    parser.add_argument('--kernel_size', default=3, type=int, help="kernel size")

    #training arguments
    parser.add_argument('--dropout', default=0, type=float, help="dropout rate")
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batchsize', default=16, type=int)
    parser.add_argument('--max_epoch', default=30, type=int, help="max epoch")
    parser.add_argument('--epoch', default=0, type=int, help="starting epoch")
    
    parser.add_argument('--folds', default=5, type=int, help = "number of folds for k-fold cv.")
    parser.add_argument('--output_directory', default='/mimer/NOBACKUP/groups/brainage/thesis_brainage/results', type=str, help="directory path for saving model and outputs")
    parser.add_argument('--run_name', default='test_run', type=str, help="name of the run")


    args = parser.parse_args()

    return args

args = parse_args()

train_participants = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/participant_files/train_fold_0.csv')
val_participants = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/participant_files/val_fold_0.csv')

train_data = loader3D(args,train_participants)
val_data = loader3D(args,val_participants)

len_train = len(train_data.demo)
len_val = len(val_data.demo)

# Collect flattened features
flattened_features_train = []
targets_train = []

for i in range(len_train):
    target = train_data.demo.iloc[i][args.target_name]
    images_and_meta = train_data[i]
    image1 = images_and_meta[0]
    image2 = images_and_meta[1]
    image_difference = image2 - image1  # shape: (1, D, H, W)

    # Flatten and convert to numpy
    flattened = image_difference.squeeze(0).flatten().numpy()
    flattened_features_train.append(flattened)
    targets_train.append(target)

X_train = np.stack(flattened_features_train)  # (n_samples, n_voxels)
y_train = np.array(targets_train).astype(np.float32)  # (n_samples,)

# Collect flattened features
flattened_features_val = []
targets_val = []

for i in range(len_val):
    target = val_data.demo.iloc[i][args.target_name]
    images_and_meta = val_data[i]
    image1 = images_and_meta[0]
    image2 = images_and_meta[1]
    image_difference = image2 - image1  # shape: (1, D, H, W)

    # Flatten and convert to numpy
    flattened = image_difference.squeeze(0).flatten().numpy()
    flattened_features_val.append(flattened)
    targets_val.append(target)

X_val = np.stack(flattened_features_val)  # (n_samples, n_voxels)
y_val = np.array(targets_val).astype(np.float32)  # (n_samples,)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_val)
mse_val = mean_squared_error(y_val, y_pred)
mae_val = np.mean(np.abs(y_val - y_pred))

# 1. Save metrics (MSE / MAE)
metrics = pd.DataFrame({
    'MSE_val': [mse_val],
    'MAE_val': [mae_val]
})
metrics.to_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/further_analysis_results/linear_regression_metrics.csv', index=False)


# 2. Save predicted vs. true targets on validation set
predictions_df = pd.DataFrame({
    'y_true': y_val,
    'y_pred': y_pred
})
predictions_df.to_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/further_analysis_results/linear_regression_val_predictions.csv', index=False)




