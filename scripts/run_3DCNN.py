from loader_CS import loader3D, load_participants
from CS_CNN import CS_CNN

import torch
import numpy as np
import os
import json
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_directory', default='/mimer/NOBACKUP/groups/brainage/data/oasis3', type=str, help="directory of the data (OASIS3)")
    parser.add_argument('--project_data_dir', default ='/mimer/NOBACKUP/groups/brainage/thesis_brainage/data', type=str, help="directory with the updated session files")
    parser.add_argument('--folds_dir', default = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/participant_files', type = str, help = 'path to participants csv.')

    #data preprocessing arguments
    parser.add_argument('--image_size', nargs=3, type=int, default=[128, 128, 128], help='Input image size as three integers (e.g. 128 128 128)')
    parser.add_argument('--image_channel', default=1, type=int, help="number of channels in the input image")

    #target and optional meta data arguments
    parser.add_argument('--target_name', default='age', type=str, help="name of the target variable")
    parser.add_argument('--optional_meta', nargs='+', default=['sex_M'], help="List of optional meta to be used in the model")
    
    #model architecture arguments
    parser.add_argument('--n_of_blocks', default=4, type=int, help="number of blocks in the encoder")
    parser.add_argument('--initial_channel', default=16, type=int, help="initial channel size after first conv")
    parser.add_argument('--kernel_size', default=3, type=int, help="kernel size")

    #training arguments
    parser.add_argument('--dropout', default=0.1, type=float, help="dropout rate")
    parser.add_argument('--epoch_weight_decay', default=15, type=int, help="epoch after which to decay the learning rate")
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batchsize', default=16, type=int)
    parser.add_argument('--max_epoch', default=20, type=int, help="max epoch")
    parser.add_argument('--epoch', default=0, type=int, help="starting epoch")
    parser.add_argument('--ignore_folds', nargs='+', default = [], help="list of folds to ignore, e.g. 0 1 2")

    parser.add_argument('--folds', default=5, type=int, help = "number of folds for k-fold cv. 0 for no cv.")
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
    """
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model, Loss, Optimizer
    model = CS_CNN(opt).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = StepLR(optimizer, step_size=opt.epoch_weight_decay, gamma=0.1)

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
        print("Epoch: ", epoch)
        model.train()
        total_loss = 0
        total_mae = 0

        for batch in dataloader_train:
            # Unpack batch
            if len(batch) == 2:
                x, target = batch
                meta = None
            else:
                x, meta, target = batch
                meta = meta.float().to(device)

            # Move tensors to device
            x = x.float().to(device)
            target = target.float().to(device)

            # Forward pass
            output = model(x, meta)
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
        avg_train_mae = total_mae / len(dataloader_train)
        train_losses.append(avg_train_loss)
        train_mae.append(avg_train_mae)
        print(f"Avg Train Loss/MAE = {avg_train_loss:.4f} / {avg_train_mae:.4f}. Weights were updated.")

        # Validation phase
        model.eval()
        total_val_loss = 0
        total_val_mae = 0

        with torch.no_grad():
            for batch in dataloader_val:
                if len(batch) == 2:
                    x, target = batch
                    meta = None
                else:
                    x, meta, target = batch
                    meta = meta.float().to(device)

                # Move tensors to device
                x = x.float().to(device)
                target = target.float().to(device)

                # Forward pass
                output = model(x, meta)
                val_loss = criterion(output, target)
                total_val_loss += val_loss.item()

                # Calculate MAE
                mae = torch.mean(torch.abs(output - target))
                total_val_mae += mae.item()

        # Log the average validation loss
        avg_val_loss = total_val_loss / len(dataloader_val)
        avg_val_mae = total_val_mae / len(dataloader_val)
        val_losses.append(avg_val_loss)
        val_mae.append(avg_val_mae)
        print(f"Avg Val Loss/MAE = {avg_val_loss:.4f} / {avg_val_mae:.4f}")

        # Scheduler step
        scheduler.step()
        print(f"Current learning rate: {scheduler.get_last_lr()[0]}")

        # Save model state
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

    return model, train_losses, train_mae, val_losses, val_mae


if __name__ == "__main__":
    # Parse command line arguments
    opt = parse_args()

    #create output directory
    output_dir = os.path.join(opt.output_directory, opt.run_name)
    os.makedirs(output_dir, exist_ok=True)

    #save details of run
    save_args_to_json(opt, os.path.join(output_dir,'run_details.json'))

    for i in range(opt.folds):

        if i in opt.ingore_folds:
            print(f"Skipping fold {i} as per ignore_folds list.")
            continue

        print("Fold:", i)
        train_fold =pd.read_csv(os.path.join(opt.folds_dir, f'train_fold_{i}.csv'))
        val_fold =pd.read_csv(os.path.join(opt.folds_dir, f'val_fold_{i}.csv'))
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

        # Plot training and validation losses
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
        print(f"Loss plot saved to: {plot_path}")

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
        print(f"MAE plot saved to: {plot_path}")