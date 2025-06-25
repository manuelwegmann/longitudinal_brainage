import torchio as tio
import pandas as pd
import os
import glob
import numpy as np
import torch
from medvae import MVAE

import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_directory', default='/mimer/NOBACKUP/groups/brainage/data/oasis3', type=str, help="directory of the data (OASIS3)")
    parser.add_argument('--project_data_dir', default ='/mimer/NOBACKUP/groups/brainage/thesis_brainage/data', type=str, help="directory with the updated session files")
    parser.add_argument('--image_path_file', default='/mimer/NOBACKUP/groups/brainage/thesis_brainage/further_analysis_results/torchio_image_dimensions.csv', type = str, help="CSV file containing all image paths.")
    parser.add_argument('--compression', default='0', type=str, help="comression factor, choose 4 or 8.")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    image_paths_csv = pd.read_csv(args.image_path_file)
    paths = image_paths_csv['path'].unique()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MVAE(model_name=f"medvae_{args.compression}_1_3d", modality="mri").to(device)
    model.requires_grad_(False)
    model.eval()

    for fpath in paths:

        dir_path = os.path.dirname(fpath)
        parts = dir_path.split(os.sep)

        sub = next((p for p in parts if p.startswith('sub-')), None)
        ses = next((p for p in parts if p.startswith('ses-')), None)
        print(sub)
        print(ses)

        img = model.apply_transform(fpath).to(device)

        with torch.no_grad():
            latent = model(img)

        spath = os.path.join(args.project_data_dir, sub, ses, f"{sub}_{ses}_latent_{args.compression}.pt")
        os.makedirs(os.path.dirname(spath), exist_ok=True)
        torch.save(latent.cpu(), spath)
