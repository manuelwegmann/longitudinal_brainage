from loader import loader3D, load_participants
from CV_LILAC import CV_LILAC

import torch
import numpy as np
import os
import json
import math
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_directory', default='/mimer/NOBACKUP/groups/brainage/data/oasis3', type=str, help="directory of the data (OASIS3)")

    #data preprocessing arguments
    parser.add_argument('--clean', default=True, type=bool, help="whether to clean data from CI and single scan participants")
    parser.add_argument('--image_size', nargs=3, type=int, default=[128, 128, 128], help='Input image size as three integers (e.g. 128 128 128)')
    parser.add_argument('--image_channel', default=1, type=int, help="number of channels in the input image")
    parser.add_argument('--val_size', default=0.2, type=float, help="validation size for splitting the data")
    parser.add_argument('--test_size', default=0.2, type=float, help="test size for splitting the data")
    parser.add_argument('--seed', default=15, type=int)

    #target and optional meta data arguments
    parser.add_argument('--target_name', default='duration', type=str, help="name of the target variable")
    parser.add_argument('--optional_meta', nargs='+', default=['sex_F', 'sex_M'], help="List of optional meta to be used in the model")
    
    #model architecture arguments
    parser.add_argument('--n_of_blocks', default=4, type=int, help="number of blocks in the encoder")
    parser.add_argument('--initial_channel', default=16, type=int, help="initial channel size after first conv")
    parser.add_argument('--kernel_size', default=3, type=int, help="kernel size")
    parser.add_argument('--conv_act', default='leaky_relu', type=str, help="activation function")
    #parser.add_argument('--pooling', default=nn.AvgPool3d, type=nn.Module, help="pooling function")

    #training arguments
    parser.add_argument('--dropout', default=0, type=float, help="dropout rate")
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batchsize', default=16, type=int)
    parser.add_argument('--max_epoch', default=30, type=int, help="max epoch")
    parser.add_argument('--epoch', default=0, type=int, help="starting epoch")
    parser.add_argument('--save_epoch_num', default=1, type=int, help="validate and save every N epoch")

    parser.add_argument('--folds', default=5, type=int, help = "number of folds for k-fold cv. 0 for no cv.")
    parser.add_argument('--output_directory', default='/mimer/NOBACKUP/groups/brainage/thesis_brainage/results', type=str, help="directory path for saving model and outputs")
    parser.add_argument('--run_name', default='test_run', type=str, help="name of the run")


    args = parser.parse_args()

    return args


def save_args_to_json(args, filepath):
    with open(filepath, 'w') as f:
        json.dump(vars(args), f, indent=4)


def split(opt, participant_df, output_dir = None):
    """
    Splits the data into training, validation and testing sets and returns the dataframes.
    Input:
        opt: options from the command line
        participant_df: dataframe with the participants and their gender
        output_dir: directory to save the datasets (set to None if not needed)
    Output:
        train_dataset: dataframe with the training set (id, sex)
        val_dataset: dataframe with the validation set (id, sex)
        test_dataset: dataframe with the testing set (id, sex)
    """
    train_dataset, temp_dataset = train_test_split(participant_df, test_size=opt.test_size + opt.val_size, random_state=opt.seed)
    test_relative_size = opt.test_size / (opt.test_size + opt.val_size)
    val_dataset, test_dataset = train_test_split(temp_dataset, test_size=test_relative_size, random_state=opt.seed)
    print(f"Train size (number of participants): {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    # Save the datasets to CSV files if output_dir is provided
    if output_dir is not None:
        train_dataset.to_csv(os.path.join(output_dir, 'train_dataset.csv'), index=False)
        val_dataset.to_csv(os.path.join(output_dir, 'val_dataset.csv'), index=False)
        test_dataset.to_csv(os.path.join(output_dir, 'test_dataset.csv'), index=False)

    return train_dataset, val_dataset, test_dataset



def train(opt, train_dataset, val_dataset):
    """
    Trains the model.
    Input:
        opt: options from the command line
        train_dataset: dataframe with the training set (id, sex)
        val_dataset: dataset with the validation set (id,sex)
    Output:
        model: trained model
        plots for training loss to output directory, as well as predicted values for train and val
    """
    # Set up device
    print("We are in train.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model, Loss, Optimizer
    model = CV_LILAC(opt).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # Data loaders
    dataloader_train = DataLoader(loader3D(opt, train_dataset), batch_size=opt.batchsize, shuffle=True)
    dataloader_val = DataLoader(loader3D(opt, val_dataset), batch_size=opt.batchsize, shuffle=False)
    
    # lists to store losses (for plot later)
    train_losses = []
    train_mae = []
    val_losses = []
    val_mae = []

    # Training loop
    for epoch in range(opt.epoch, opt.max_epoch):
        print("We are in epoch (training): ", epoch)
        model.train()
        total_loss = 0
        total_mae = 0

        for batch in dataloader_train:
            # Unpack batch
            if len(batch) == 3:
                x1, x2, target = batch
                meta = None
            else:
                x1, x2, meta, target = batch
                meta = meta.float().to(device)

            # Move tensors to device
            x1 = x1.float().to(device)
            x2 = x2.float().to(device)
            target = target.float().to(device)

            # Forward pass
            output = model(x1, x2, meta)
            loss = criterion(output, target)

            #calculate MAE
            mae = torch.mean(torch.abs(output - target))
            total_mae += mae.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Log the average training loss
        avg_train_loss = total_loss / len(dataloader_train)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch}: Avg Train Loss = {avg_train_loss:.4f}")

        # Output MAE
        avg_train_mae = total_mae / len(dataloader_train)
        train_mae.append(avg_train_mae)
        print(f"Epoch {epoch}: Avg Train MAE = {avg_train_mae:.4f}")

        # Validation phase
        model.eval()
        print("We are in epoch (val): ", epoch)
        total_val_loss = 0
        total_val_mae = 0

        with torch.no_grad():
            for batch in dataloader_val:
                if len(batch) == 3:
                    x1, x2, target = batch
                    meta = None
                else:
                    x1, x2, meta, target = batch
                    meta = meta.float().to(device)

                # Move tensors to device
                x1 = x1.float().to(device)
                x2 = x2.float().to(device)
                target = target.to(device).float()

                # Forward pass
                output = model(x1, x2, meta)
                val_loss = criterion(output, target)
                total_val_loss += val_loss.item()

                # Calculate MAE
                mae = torch.mean(torch.abs(output - target))
                total_val_mae += mae.item()

        # Log the average validation loss
        avg_val_loss = total_val_loss / len(dataloader_val)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch}: Avg Val Loss = {avg_val_loss:.4f}")

        # Output MAE
        avg_val_mae = total_val_mae / len(dataloader_val)
        val_mae.append(avg_val_mae)
        print(f"Epoch {epoch}: Avg Val MAE = {avg_val_mae:.4f}")

        model_state = model.state_dict()  # Save model weights
        model_path = os.path.join(opt.output_directory, opt.run_name, 'model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss
        }, model_path)
            

    # Calculate predicted values on train and val set
    if model_state is not None:
        model.load_state_dict(model_state)
        print("Training complete. Will now calculate predicted values on train and test set.")

        model.eval()

        train_preds = []
        train_targets = []

        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in dataloader_train:
                if len(batch) == 3:
                    x1, x2, target = batch
                    meta = None
                else:
                    x1, x2, meta, target = batch
                    meta = meta.float().to(device)

                x1 = x1.float().to(device)
                x2 = x2.float().to(device)
                target = target.to(device).float()

                output = model(x1, x2, meta)

                train_preds.append(output.cpu())
                train_targets.append(target.cpu())

            for batch in dataloader_val:
                if len(batch) == 3:
                    x1, x2, target = batch
                    meta = None
                else:
                    x1, x2, meta, target = batch
                    meta = meta.float().to(device)

                x1 = x1.float().to(device)
                x2 = x2.float().to(device)
                target = target.to(device).float()

                output = model(x1, x2, meta)

                val_preds.append(output.cpu())
                val_targets.append(target.cpu())

            
        # Save predictions and targets for train and val to CSV
        targets = np.concatenate(train_targets, axis=0)
        preds = np.concatenate(train_preds, axis=0)
        results_df = pd.DataFrame({
            "Target (Train)": targets.flatten(),
            "Prediction (Train)": preds.flatten()
        })
        results_path = os.path.join(opt.output_directory, opt.run_name, "train_predicted_values.csv")
        results_df.to_csv(results_path, index=False)

        targets = np.concatenate(val_targets, axis=0)
        preds = np.concatenate(val_preds, axis=0)
        results_df = pd.DataFrame({
            "Target (Val)": targets.flatten(),
            "Prediction (Val)": preds.flatten()
        })
        results_path = os.path.join(opt.output_directory, opt.run_name, "val_predicted_values.csv")
        results_df.to_csv(results_path, index=False)
    

    # Shift validation losses forward by one epoch for alignment
    val_losses = [np.nan] + val_losses[:-1]
    val_mae = [np.nan] + val_mae[:-1]

    # Plot and save training/validation loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(opt.output_directory, opt.run_name, 'loss_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss plot (Train/Val) saved to: {plot_path}")

    # Plot and save training/validation mae curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_mae, label='Training MAE')
    plt.plot(val_mae, label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (years)')
    plt.title('Training and Validation Mean Absolute Error')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(opt.output_directory, opt.run_name, 'mae_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"MAE plot (Train/Val) saved to: {plot_path}")

    return model



if __name__ == "__main__":
    print("We are in main.")

    # Parse command line arguments
    opt = parse_args()
    save_runname = opt.run_name

    #create output directory
    output_dir = os.path.join(opt.output_directory, opt.run_name)
    os.makedirs(output_dir, exist_ok=True)

    #save details of run
    save_args_to_json(opt, os.path.join(output_dir,'run_details.json'))

    # Setup data
    participant_df = load_participants(folder_path = opt.data_directory, clean = opt.clean, add_age = True)
        
    age_values = participant_df['age'].values 

    # Bin the targets into quantile-based bins
    binned_age = pd.qcut(age_values, q=10, labels=False)
    
    # Create stratified K-folds using the binned targets
    skf = StratifiedKFold(n_splits=opt.folds, shuffle=True, random_state=opt.seed)
    folds = list(skf.split(participant_df, binned_age))

    for i, (train_idx, val_idx) in enumerate(folds):
        print("We are in fold:", i)
        train_fold = participant_df.iloc[train_idx].reset_index(drop=True)
        val_fold = participant_df.iloc[val_idx].reset_index(drop=True)
        train_fold = train_fold[['participant_id', 'sex']]
        val_fold = val_fold[['participant_id', 'sex']]

        # Set output directory and save CSVs
        opt.output_directory = output_dir
        opt.run_name = f"fold_{i}"
        os.makedirs(os.path.join(opt.output_directory, opt.run_name), exist_ok=True)

        train_fold.to_csv((os.path.join(opt.output_directory, opt.run_name, 'train_fold.csv')), index=False)
        val_fold.to_csv((os.path.join(opt.output_directory, opt.run_name, 'val_fold.csv')), index=False)

        # Train and test
        trained_model = train(opt, train_fold, val_fold)