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