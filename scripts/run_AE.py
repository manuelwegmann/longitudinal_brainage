from loader_AE import loader3D
import numpy as np
import os
import argparse
import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_directory', default='/mimer/NOBACKUP/groups/brainage/data/oasis3', type=str, help="directory of the data (OASIS3)")
    parser.add_argument('--project_data_dir', default ='/mimer/NOBACKUP/groups/brainage/thesis_brainage/data', type=str, help="directory with the updated session files")
    parser.add_argument('--folds_dir', default = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds', type = str, help = 'path to participants csv.')

    #data preprocessing arguments
    parser.add_argument('--compression', default=0, type=int, help='compression used in autoencoder (4 or 8)')
    parser.add_argument('--image_channel', default=1, type=int, help="number of channels in the input image")

    #target and optional meta data arguments
    parser.add_argument('--target_name', default='duration', type=str, help="name of the target variable")
    parser.add_argument('--optional_meta', nargs='+', default=['sex_M'], help="List of optional meta to be used in the model")
    parser.add_argument('--folds', default=5, type=int, help="number of folds.")
    parser.add_argument('--ignore_folds', nargs='+', default = [], help="list of folds to ignore, e.g. 0 1 2")
    
    #results
    
    parser.add_argument('--output_directory', default='/mimer/NOBACKUP/groups/brainage/thesis_brainage/results', type=str, help="directory path for saving model and outputs")
    parser.add_argument('--run_name', default='test_run', type=str, help="name of the run")


    args = parser.parse_args()

    return args

def save_args_to_json(args, filepath):
    with open(filepath, 'w') as f:
        json.dump(vars(args), f, indent=4)


if __name__ == "__main__":
    # Parse command line arguments
    opt = parse_args()

    #create output directory
    output_dir = os.path.join(opt.output_directory, opt.run_name)
    os.makedirs(output_dir, exist_ok=True)

    #save details of run
    save_args_to_json(opt, os.path.join(output_dir,'run_details.json'))

    for i in range(opt.folds):

        if i in opt.ignore_folds:
            print(f"Skipping fold {i} as per ignore_folds list.")
            continue

        print("Fold:", i)

        #Load data
        train_fold = pd.read_csv(os.path.join(opt.folds_dir, f'train_fold_{i}.csv'))
        train_fold = train_fold[['participant_id', 'sex']]
        train_data = loader3D(opt,train_fold)
        n_train = len(train_data.demo)

        val_fold = pd.read_csv(os.path.join(opt.folds_dir, f'val_fold_{i}.csv'))
        val_fold = val_fold[['participant_id', 'sex']]
        val_data = loader3D(opt,val_fold)
        n_val = len(val_data.demo)

        # Set output directory and save CSVs
        opt.output_directory = output_dir
        opt.run_name = f"fold_{i}"
        os.makedirs(os.path.join(opt.output_directory, opt.run_name), exist_ok=True)

        train_fold.to_csv((os.path.join(opt.output_directory, opt.run_name, 'train_fold.csv')), index=False)
        val_fold.to_csv((os.path.join(opt.output_directory, opt.run_name, 'val_fold.csv')), index=False)

        """
        Prep data
        """
        flattened_features_train = []
        targets_train = []

        for j in range(n_train):
            image1, image2, meta, target = train_data[j]

            image_difference = image2 - image1
            brain_features = image_difference.squeeze(0).flatten().numpy()

            meta = meta.numpy()
            meta = meta.flatten() 

            target = target.numpy()

            features = np.concatenate([brain_features, meta])

            flattened_features_train.append(features)
            targets_train.append(target)

        X_train = np.stack(flattened_features_train)
        y_train = np.array(targets_train).astype(np.float32)

        print("Prepared training data.")


        flattened_features_val = []
        targets_val = []

        for j in range(n_val):
            image1, image2, meta, target = val_data[j]

            image_difference = image2 - image1
            brain_features = image_difference.squeeze(0).flatten().numpy()

            meta = meta.numpy()
            meta = meta.flatten()

            target = target.numpy()

            features = np.concatenate([brain_features, meta])

            flattened_features_val.append(features)
            targets_val.append(target)

        X_val = np.stack(flattened_features_val)
        y_val = np.array(targets_val).astype(np.float32)

        print("Prepared validation data.")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)  # Fit on train
        X_val = scaler.transform(X_val)          # Apply same scaling to va

        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        y_train_pred = regressor.predict(X_train)
        y_val_pred = regressor.predict(X_val)

        MSE_train = mean_squared_error(y_train_pred, y_train)
        MAE_train = mean_absolute_error(y_train_pred, y_train)

        MSE_val = mean_squared_error(y_val_pred, y_val)
        MAE_val = mean_absolute_error(y_val_pred, y_val)

        metrics = pd.DataFrame({
            "MSE Train": [MSE_train],
            "MAE Train": [MAE_train],
            "MSE Val": [MSE_val],
            "MAE Val": [MAE_val]
        })

        csv_path = os.path.join(opt.output_directory, opt.run_name, 'training_metrics.csv')
        metrics.to_csv(csv_path, index=False)

        results = pd.DataFrame({
            "Participant_ID": val_data.demo['participant_id'],
            "Target": y_val.ravel(),
            "Prediction": y_val_pred.ravel(),
            "Age": val_data.demo['age'],
            "Sex (M)": val_data.demo['sex_M'],
            "Session 1": val_data.demo['session_id1'],
            "Session 2": val_data.demo['session_id2']
        })

        csv_path = os.path.join(opt.output_directory, opt.run_name, 'predictions_val.csv')
        results.to_csv(csv_path, index=False)

    path_to_folds = opt.output_directory
    dfs = []

    # Iterate through each item in the directory
    for folder_name in os.listdir(path_to_folds):
        if folder_name.startswith("fold_"):
            folder_path = os.path.join(path_to_folds, folder_name)
            csv_path = os.path.join(folder_path, "predictions_val.csv")
            
            if os.path.isfile(csv_path):
                df = pd.read_csv(csv_path)
                df['fold'] = folder_name  # Optionally tag the fold
                dfs.append(df)

    # Combine all DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)

    # Preview the result
    combined_df.to_csv(os.path.join(path_to_folds, 'predictions_all_folds.csv'))

        