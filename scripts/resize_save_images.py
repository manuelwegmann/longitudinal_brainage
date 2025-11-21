from loader import loader3D
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

args = load_args_from_json('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC/run_details.json')

participants1 = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/train_fold_0.csv')
participants2 = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/val_fold_0.csv')

data1 = loader3D(args,participants1).demo
data2 = loader3D(args,participants2).demo
data = pd.concat([data1, data2], ignore_index=True)
print(data.head())

transformation = tio.transforms.Resize(tuple(args.image_size))

for _,row in data.iterrows():

    p_id = row['participant_id']
    s1_id = row['session_id1']
    s2_id = row['session_id2']
    img_dir1 = os.path.join('/mimer/NOBACKUP/groups/brainage/data/oasis3/derivatives/mriprep', p_id, s1_id)
    img_dir2 = os.path.join('/mimer/NOBACKUP/groups/brainage/data/oasis3/derivatives/mriprep', p_id, s2_id)
    pattern1 = os.path.join(img_dir1, '*T1w.nii.gz')
    pattern2 = os.path.join(img_dir2, '*T1w.nii.gz')

    matching_files1 = glob.glob(pattern1)
    matching_files2 = glob.glob(pattern2)

    if not matching_files1 or not matching_files2: #skip if no matching files are found
        print(f"Warning: No matching T1w image found for {p_id} in session(s). Skipping.")
        continue
    path1 = matching_files1[0]
    path2 = matching_files2[0]

    output_path1 = os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/resized_images', f'{p_id}_{s1_id}_resized.pt')
    if not os.path.exists(output_path1):
        image1 = tio.ScalarImage(path1)
        image1 = transformation(image1)
        image1_tensor = image1.data
        torch.save(image1_tensor, output_path1)
        print(f'Saved resized image to {output_path1}')

    output_path2 = os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/resized_images', f'{p_id}_{s2_id}_resized.pt')
    if not os.path.exists(output_path2):
        image2 = tio.ScalarImage(path2)
        image2 = transformation(image2)
        image2_tensor = image2.data
        torch.save(image2_tensor, output_path2)
        print(f'Saved resized image to {output_path2}')
    
