import torch
import os
from medvae import MVAE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fpath = "/mimer/NOBACKUP/groups/brainage/data/oasis3/derivatives/mriprep/sub-OAS30001/ses-d0129/sub-OAS30001_ses-d0129_space-MNI152NLin2009cAsym_desc-brain_T1w.nii.gz"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MVAE(model_name="medvae_4_1_3d", modality="mri").to(device)
print("model loaded, will now be applied.")
img = model.apply_transform(fpath).to(device)

model.requires_grad_(False)
model.eval()

print("Check.")
with torch.no_grad():
    decoded_img, latent = model(img, decode=True)

print("Ran model. Now plotting slices.")

# Get axial slice (from top of the head)
slice_idx1 = img.shape[2] // 2
slice_idx2 = latent.shape[2] // 2

# Move tensors to CPU and convert to NumPy
slice_original = img[:, :, slice_idx1].cpu().numpy()
slice_latent = latent[:, :, slice_idx2].cpu().numpy()
slice_decoded = decoded_img[:, :, slice_idx1].cpu().numpy()

# If needed, squeeze singleton dimensions (e.g., [1, 160, 192] → [160, 192])
slice_original = np.squeeze(slice_original)
slice_latent = np.squeeze(slice_latent)
slice_decoded = np.squeeze(slice_decoded)

# Define save paths
save_dir = "/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/plots"
os.makedirs(save_dir, exist_ok=True)
save_path1 = os.path.join(save_dir, "original_axial.png")
save_path2 = os.path.join(save_dir, "latent_axial.png")
save_path3 = os.path.join(save_dir, "decoded_axial.png")

# Plot and save Original Image
plt.figure(figsize=(5, 5))
plt.imshow(slice_original.T, cmap='gray', origin='lower')
plt.title("Original Image")
plt.axis('off')
plt.savefig(save_path1, bbox_inches='tight', dpi=300)
plt.close()

# Plot and save Latent Representation
plt.figure(figsize=(5, 5))
plt.imshow(slice_latent.T, cmap='gray', origin='lower')
plt.title("Latent Representation")
plt.axis('off')
plt.savefig(save_path2, bbox_inches='tight', dpi=300)
plt.close()

# Plot and save Decoded Image
plt.figure(figsize=(5, 5))
plt.imshow(slice_decoded.T, cmap='gray', origin='lower')
plt.title("Decoded Image")
plt.axis('off')
plt.savefig(save_path3, bbox_inches='tight', dpi=300)
plt.close()

print(f"Saved plots to:\n{save_path1}\n{save_path2}\n{save_path3}")