import torchio as tio
import pandas as pd
from pathlib import Path
import os
import glob

import argparse
from argparse import Namespace
import json

from loader_fs import load_participants, loader3D

def load_args_from_json(filepath):

    with open(filepath, 'r') as f:
        args_dict = json.load(f)

    args = Namespace(**args_dict)

    return args

opt = load_args_from_json('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/fs_LILAC_plus/run_details.json')

participants_df = load_participants()
data = loader3D(opt, participants_df).demo
participant_ids = []
scan_ids = []

for _, row in data.iterrows():
    pid = str(row['participant_id'])
    s1id = str(row['session_id1'])
    s2id = str(row['session_id2'])
    participant_ids.append(pid)
    participant_ids.append(pid)
    scan_ids.append(s1id)
    scan_ids.append(s2id)
    
image_paths = []

#/mimer/NOBACKUP/groups/brainage/data/oasis3/derivatives/mriprep/sub-OAS30900/ses-d0647/sub-OAS30900_ses-d0647_space-MNI152NLin2009cAsym_desc-brain_T1w.nii.gz

for i in range(len(participant_ids)):

    image_folder = os.path.join('/mimer/NOBACKUP/groups/brainage/data/oasis3/derivatives/mriprep', participant_ids[i], scan_ids[i])

    # Recursive search
    t1w_files = glob.glob(os.path.join(image_folder, '*T1w.nii.gz'))

    if t1w_files:
        first_t1w_path = t1w_files[0]
        print("Found:", first_t1w_path)
        image_paths.append(first_t1w_path)
    else:
        print("No T1w file found in:", image_folder)

# Store the results
records = []

for path in image_paths:
    try:
        img = tio.ScalarImage(path)
        tensor_shape = img.data.shape  # shape is (C, D, H, W)
        spacing = img.spacing  # (Z, Y, X)
        records.append({
            'filename': Path(path).name,
            'path': path,
            'channels': tensor_shape[0],
            'depth': tensor_shape[1],
            'height': tensor_shape[2],
            'width': tensor_shape[3],
            'spacing_z': spacing[0],
            'spacing_y': spacing[1],
            'spacing_x': spacing[2],
        })
    except Exception as e:
        print(f"Error with {path}: {e}")
        records.append({
            'filename': Path(path).name,
            'path': path,
            'error': str(e)
        })

# Save or preview the result
df = pd.DataFrame(records)
df.to_csv("/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/torchio_image_dimensions.csv", index=False)
print(df.head())