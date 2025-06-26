from loader_fs import loader3D, load_participants
from LILAC import LILAC
from LILAC_plus import LILAC_plus


import numpy as np
import os
import json
import pandas as pd
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_directory', default='/mimer/NOBACKUP/groups/brainage/data/oasis3', type=str, help="directory of the data (OASIS3)")
    parser.add_argument('--project_data_dir', default ='/mimer/NOBACKUP/groups/brainage/thesis_brainage/data', type=str, help="directory with the updated session files")

    parser.add_argument('--model', default='LILAC_plus', type=str, choices=['LILAC', 'LILAC_plus'], help="model to use: LILAC or LILAC_plus")

    #data preprocessing arguments
    parser.add_argument('--compression', default=0, type=int, help='compression used in autoencoder (4 or 8)')
    parser.add_argument('--image_channel', default=1, type=int, help="number of channels in the input image")
    parser.add_argument('--seed', default=15, type=int)

    #target and optional meta data arguments
    parser.add_argument('--target_name', default='duration', type=str, help="name of the target variable")
    parser.add_argument('--optional_meta', nargs='+', default=['sex_F', 'sex_M'], help="List of optional meta to be used in the model")
    
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


def save_args_to_json(args, filepath):
    with open(filepath, 'w') as f:
        json.dump(vars(args), f, indent=4)



def train(opt, train_dataset, val_dataset):
    """
    Trains the model.
    Input:
        opt: options from the command line
        train_dataset: dataframe with the training set (id, sex)
        val_dataset: dataset with the validation set (id,sex)
    Output:
        model: trained model
        training metrics: lists of training and validation losses and MAE
        plots for training and validation loss to output directory.
    """
    # Set up device
    print("We are in train.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model, Loss, Optimizer
    if opt.model == 'LILAC':
        model = LILAC(opt).to(device)
    elif opt.model == 'LILAC_plus':
        model = LILAC_plus(opt).to(device)
    else:
        raise ValueError("Invalid model type. Choose 'LILAC' or 'LILAC_plus'.")
    
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
                target = target.float().to(device)

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

    return model, train_losses, train_mae, val_losses, val_mae



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
    participant_df = load_participants(project_data_dir = opt.project_data_dir, folder_path = opt.data_directory, add_age = True)
        
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

        # Train
        trained_model, train_losses, train_mae, val_losses, val_mae = train(opt, train_fold, val_fold)

        training_metrics = pd.DataFrame({
            'train_loss': train_losses,
            'train_mae': train_mae,
            'val_loss': val_losses,
            'val_mae': val_mae
        })

        # Define directory and file path
        csv_path = os.path.join(opt.output_directory, opt.run_name, 'training_metrics.csv')
        training_metrics.to_csv(csv_path, index=False)


        