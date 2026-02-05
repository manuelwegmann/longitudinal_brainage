from loader_CS import loader3D
import os
import pandas as pd
import argparse
from argparse import Namespace
import json
import glob
import torchio as tio
import numpy as np
import torch

#function to load model parameters
def load_args_from_json(filepath):

    with open(filepath, 'r') as f:
        args_dict = json.load(f)

    args = Namespace(**args_dict)

    return args

args = load_args_from_json('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/CS_CNN/run_details.json')

participants1 = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/CI_CN_groups/CI_participants.csv')
participants2 = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/cs_val_fold_0.csv')

data1 = loader3D(args,participants1).demo
data2 = loader3D(args,participants2).demo
data = pd.concat([data1, data2], ignore_index=True)
print(data.head())

transformation = tio.transforms.Resize(tuple(args.image_size))

for _,row in data.iterrows():

    p_id = row['participant_id']
    s_id = row['session_id']
    img_dir = os.path.join('/mimer/NOBACKUP/groups/brainage/data/oasis3/derivatives/mriprep', p_id, s_id)
    pattern = os.path.join(img_dir, '*T1w.nii.gz')

    matching_files = glob.glob(pattern)

    if not matching_files: #skip if no matching files are found
        print(f"Warning: No matching T1w image found for {p_id} in session {s_id}). Skipping.")
        continue
    path = matching_files[0]

    output_path = os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/resized_images', f'{p_id}_{s_id}_resized.pt')
    if not os.path.exists(output_path):
        image = tio.ScalarImage(path)
        image = transformation(image)
        image_tensor = image.data
        torch.save(image_tensor, output_path)
        print(f'Saved resized image to {output_path}')