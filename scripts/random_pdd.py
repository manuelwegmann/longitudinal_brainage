"""
Script to train LILAC for longitudinal brain age task with progressive data dropout.
"""

from new_loader import loader3D
from LILAC import LILAC

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
from torch.optim.lr_scheduler import StepLR


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_directory', default='/mimer/NOBACKUP/groups/brainage/data/oasis3', type=str, help="directory of the data (OASIS3)")
    parser.add_argument('--project_data_dir', default ='/mimer/NOBACKUP/groups/brainage/thesis_brainage/data', type=str, help="directory with the updated session files")
    parser.add_argument('--folds_dir', default = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds', type = str, help = 'path to participants csv.')

    #data preprocessing arguments
    parser.add_argument('--image_size', nargs=3, type=int, default=[128, 128, 128], help='Input image size as three integers (e.g. 128 128 128)')
    parser.add_argument('--image_channel', default=1, type=int, help="number of channels in the input image")

    #target and optional meta data arguments
    parser.add_argument('--target_name', default='duration', type=str, help="name of the target variable")
    parser.add_argument('--optional_meta', nargs='+', default=['sex_M'], help="List of optional meta to be used in the model")
    
    #model architecture arguments
    parser.add_argument('--n_of_blocks', default=4, type=int, help="number of blocks in the encoder")
    parser.add_argument('--initial_channel', default=16, type=int, help="initial channel size after first conv")
    parser.add_argument('--kernel_size', default=3, type=int, help="kernel size")

    #training arguments
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batchsize', default=24, type=int)
    parser.add_argument('--max_epoch', default=20, type=int, help="max epoch")
    parser.add_argument('--epoch', default=0, type=int, help="starting epoch")
    parser.add_argument('--dropout', default=0, type=float, help="dropout rate")
    
    parser.add_argument('--output_directory', default='/mimer/NOBACKUP/groups/brainage/thesis_brainage/results', type=str, help="directory path for saving model and outputs")
    parser.add_argument('--run_name', default='test_run', type=str, help="name of the run")

    parser.add_argument('--prob', default=1, type=float, help="prob of including data in backprop ( 1 = no selective backprop).")

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
    model = LILAC(opt).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # Data loaders
    dataloader_train = DataLoader(loader3D(opt, train_dataset), batch_size=opt.batchsize, shuffle=True, num_workers=12, pin_memory=True)
    dataloader_val = DataLoader(loader3D(opt, val_dataset), batch_size=opt.batchsize, shuffle=False, num_workers=12, pin_memory=True)
    
    eff_epochs = 0
    N = len(loader3D(opt, train_dataset).demo)

    p = opt.prob

    train_losses, val_losses, train_mae, val_mae, ee_per_epoch = [], [], [], [], []

    # Training loop
    for epoch in range(opt.epoch, opt.max_epoch):
        print("Epoch: ", epoch)
        model.train()
        total_loss = 0
        total_mae = 0

        datapoints_for_backprop = 0

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

            # Compute per-sample absolute error
            abs_err = torch.abs(output - target)

            # Apply selective backprop PDD
            abs_err = torch.abs(output - target)


            if epoch == opt.max_epoch - 1:
                # Last epoch: use all samples
                mask = torch.ones_like(abs_err, dtype=torch.bool)
            else:
                # Random subset selection
                mask = torch.rand_like(abs_err, dtype=torch.float32) < p**epoch

                # Guarantee at least TWO samples
                num_selected = mask.sum().item()
                if num_selected < 2:
                    needed = 2 - num_selected
                    batch_size = len(mask)
                    random_indices = torch.randperm(batch_size)[:needed]
                    for idx in random_indices:
                        mask[idx] = True

            datapoints_for_backprop += mask.sum().item()

            loss = criterion(output[mask], target[mask])

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #calculate MAE
            mae = torch.mean(abs_err)
            total_mae += mae.item()

            total_loss += loss.item()

        eff_epoch =  datapoints_for_backprop / N
        print(f"Effective Epoch here: {eff_epoch:.2f}")
        eff_epochs += eff_epoch
        print(f"Total Effective Epochs so far: {eff_epochs:.2f}")
        ee_per_epoch.append(eff_epoch)
        

        # Log the average training loss and MAE
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

        # Log the average validation loss/MAE
        avg_val_loss = total_val_loss / len(dataloader_val)
        avg_val_mae = total_val_mae / len(dataloader_val)
        val_losses.append(avg_val_loss)
        val_mae.append(avg_val_mae)
        print(f"Avg Val Loss/MAE = {avg_val_loss:.4f} / {avg_val_mae:.4f}")

        # Save model weights
        model_state = model.state_dict() 
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

    return model, train_losses, train_mae, val_losses, val_mae, ee_per_epoch



if __name__ == "__main__":
    # Parse command line arguments
    opt = parse_args()

    #create output directory
    output_dir = os.path.join(opt.output_directory, opt.run_name)
    os.makedirs(output_dir, exist_ok=True)

    #save details of run
    save_args_to_json(opt, os.path.join(output_dir,'run_details.json'))

    print("Hardcoded folds 0.")
    train_fold =pd.read_csv(os.path.join(opt.folds_dir, f'train_fold_0.csv'))
    val_fold =pd.read_csv(os.path.join(opt.folds_dir, f'val_fold_0.csv'))
    train_fold = train_fold[['participant_id', 'sex']]
    val_fold = val_fold[['participant_id', 'sex']]

    #Save folds
    train_fold.to_csv(os.path.join(output_dir, 'train_fold.csv'), index=False)
    val_fold.to_csv(os.path.join(output_dir, 'val_fold.csv'), index=False)

    # Train
    trained_model, train_losses, train_mae, val_losses, val_mae, ee_per_epoch = train(opt, train_fold, val_fold)

    training_metrics = pd.DataFrame({
        'train_loss': train_losses,
        'train_mae': train_mae,
        'val_loss': val_losses,
        'val_mae': val_mae,
        'effective_epochs': ee_per_epoch
    })

    # Define directory and file path
    csv_path = os.path.join(output_dir, 'training_metrics.csv')
    training_metrics.to_csv(csv_path, index=False)

    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, 'loss_plot.png')
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
    plot_path = os.path.join(output_dir, 'mae_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"MAE plot saved to: {plot_path}")


    # Plot effective epochs per actual epoch
    epochs = list(range(len(ee_per_epoch)))

    plt.figure(figsize=(6,4))
    plt.plot(epochs, ee_per_epoch, marker='o')
    plt.xlabel("Actual Epoch")
    plt.ylabel("Effective Epoch")
    plt.title("Effective Epochs per Actual Epoch")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'ee_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"EE plot saved to: {plot_path}")