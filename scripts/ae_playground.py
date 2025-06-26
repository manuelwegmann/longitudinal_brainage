import torch
import os
from medvae import MVAE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchio as tio

fpath = "/mimer/NOBACKUP/groups/brainage/thesis_brainage/data/sub-OAS30001/ses-d0757/sub-OAS30001_ses-d0757_latent_8.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load raw 3D volume
tensor = torch.load(fpath) 

print(tensor.shape)

tensor = tensor.unsqueeze(0)

print(tensor.shape)

# Create dummy affine (identity or your known affine)
affine = torch.eye(4)

# Wrap in ScalarImage
image = tio.ScalarImage(tensor=tensor, affine=affine)

# Resize to desired shape (e.g., [128, 128, 128])
resized = tio.Resize((24, 24, 24))(image)

print(resized.shape)

resized = resized.data

print(resized.shape)