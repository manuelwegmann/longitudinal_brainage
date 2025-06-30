import torch
import os
from medvae import MVAE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchio as tio
import argparse

from loader_AE import loader3D

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

if __name__ == '__main__':
    args = parse_args()
    participants = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/further_analysis_results/participants_file.csv')
    a = loader3D(args, participants)
    ad = a.demo
    print(ad.head())

    print(ad.iloc[49])
    b = a[49]
    print(ad.iloc[125])
    c = a[125]


import torch
import matplotlib.pyplot as plt
import os

# Load latent tensor
latent_path = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/further_analysis_results/latent.pt'
latent = torch.load(latent_path).cpu()

# Check shape
print("Latent shape:", latent.shape)  # Expecting something like (D, H, W)

# Define output folder
output_folder = "/mimer/NOBACKUP/groups/brainage/thesis_brainage/results"

# Function to plot and save a slice
def save_slice(slice_2d, orientation, index):
    plt.imshow(slice_2d, cmap='gray')
    plt.axis('off')
    fname = f"{output_folder}/slice_{orientation}_{index}.png"
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved: {fname}")

# Plot middle slices in each orientation
depth, height, width = latent.shape[-3:]

# Axial (xy) slice at middle of z
#save_slice(latent[depth // 2, :, :], 'axial', depth // 2)

# Coronal (xz) slice at middle of y
#save_slice(latent[:, height // 2, :], 'coronal', height // 2)

# Sagittal (yz) slice at middle of x
#save_slice(latent[:, :, width // 2], 'sagittal', width // 2)