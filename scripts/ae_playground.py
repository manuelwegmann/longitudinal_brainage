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

print("Ran model. Now saving latent representation.")

torch.save(latent.cpu(), '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/latent.pt')
