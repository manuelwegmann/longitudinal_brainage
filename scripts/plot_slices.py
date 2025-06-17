import pandas as pd
import os
import numpy as np
import json
from loader import loader3D
from argparse import Namespace
import matplotlib.pyplot as plt
import torch

#function to load model parameters
def load_args_from_json(filepath):

    with open(filepath, 'r') as f:
        args_dict = json.load(f)

    args = Namespace(**args_dict)

    return args

participants_file = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_age/fold_0/val_fold.csv')
args = load_args_from_json('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_age/run_details.json')
data = loader3D(args, participants_file)


# Extract one sample (first image pair)
print(data.demo.iloc[55])
sample = data[55]
if len(sample) == 4:
    image1_tensor, image2_tensor, meta, target = sample
else:
    image1_tensor, image2_tensor, target = sample

print(target)

# Remove channel dimension
img1 = image1_tensor.squeeze(0)  # shape: (D, H, W)
img2 = image2_tensor.squeeze(0)

# Get axial slice (from top of the head)
slice_idx = img1.shape[2] // 2
slice1 = img1[:, :, slice_idx]
slice2 = img2[:, :, slice_idx]

# Define save paths
save_dir = "/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/plots"
os.makedirs(save_dir, exist_ok=True)
save_path1 = os.path.join(save_dir, "image1_axial.png")
save_path2 = os.path.join(save_dir, "image2_axial.png")

# Plot and save image 1
plt.figure(figsize=(5, 5))
plt.imshow(slice1.T, cmap='gray', origin='lower')
plt.title("Baseline Image (3D)")
plt.axis('off')
plt.savefig(save_path1, bbox_inches='tight', dpi=300)
plt.close()

# Plot and save image 2
plt.figure(figsize=(5, 5))
plt.imshow(slice2.T, cmap='gray', origin='lower')
plt.title("Follow-Up Image (3D)")
plt.axis('off')
plt.savefig(save_path2, bbox_inches='tight', dpi=300)
plt.close()

print(f"Saved plots to:\n{save_path1}\n{save_path2}")