"""
This script applies a trained cross-sectional model to a .csv file containing participant ids.
"""

import numpy as np
import pandas as pd
import os
import argparse
from argparse import Namespace
import json
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

from CS_3DCNN import CS_CNN
from new_loader_CS import loader3D

#options from the command line
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--json', default='blank', type=str, help = "json file with run details.")
    parser.add_argument('--participants_file', default = 'blank', type=str, help = "participants file csv")
    parser.add_argument('--model_state', default = 'blank', type=str, help = "path to model state")

    args = parser.parse_args()

    return args

#function to load model parameters
def load_args_from_json(filepath):

    with open(filepath, 'r') as f:
        args_dict = json.load(f)

    args = Namespace(**args_dict)

    return args

#function to decide what type of data we are looking at
def decide_type_of_result(filepath):
    print("Filename (results): ", filepath)
    if 'train_fold' in filepath.lower():
        name = 'train'
    elif 'val_fold' in filepath.lower():
        name = 'val'
    elif 'test_fold' in filepath.lower():
        name = 'test'
    elif 'ci_participants' in filepath.lower():
        name = 'ci'
    elif 'cn_participants' in filepath.lower():
        name = 'cn'
    else:
        print("Error in naming of file.")
        name = 'error'
    print(f"We are looking at {name} data.")
    return name


def apply_model(opt, model, participants_df, name):
    """
    Applies a trained model to a dataset and returns predictions with metadata.
    Input:
        opt: options from the command line
        model: trained model
        participants_df: dataframe with the data
    Output:
        prints test loss and MAE
        returns a DataFrame with predictions and metadata
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataloader = DataLoader(loader3D(opt, participants_df), batch_size=opt.batchsize, shuffle=False)

    criterion = nn.MSELoss()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    all_ids = []
    all_sex_M = []
    all_sessions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if len(batch) == 2:
                x, target = batch
                meta = None
            else:
                x, meta, target = batch
                meta = meta.to(device)

            x = x.to(device)
            target = target.to(device)

            output = model(x, meta)
            loss = criterion(output, target)
            total_loss += loss.item()

            all_targets.append(target.cpu().numpy())
            all_preds.append(output.cpu().numpy())

            start_idx = batch_idx * opt.batchsize
            end_idx = start_idx + x.size(0)
            batch_demo = dataloader.dataset.demo.iloc[start_idx:end_idx]

            all_ids.extend(batch_demo["participant_id"].tolist())
            all_sex_M.extend(batch_demo["sex_M"].tolist())
            all_sessions.extend(batch_demo["session_id"].tolist())

    avg_loss = total_loss / len(dataloader)
    print(f"{name} Loss (MSE): {avg_loss:.4f}")

    # Convert predictions and targets
    targets = np.concatenate(all_targets, axis=0).flatten()
    preds = np.concatenate(all_preds, axis=0).flatten()

    mae = np.mean(np.abs(preds - targets))
    print(f"{name} MAE: {mae:.4f}")

    # Build results DataFrame
    results = pd.DataFrame({
        "Participant_ID": all_ids,
        "Target": targets,
        "Prediction": preds,
        "Sex (M)": all_sex_M,
        "Sessions": all_sessions
    })

    return results



if __name__ == "__main__":

    args = parse_args()

    opt = load_args_from_json(args.json)

    name = decide_type_of_result(args.participants_file)

    print(f"Model saved at {args.model_state} will be applied to participants at {args.participants_file}.")

    participants_df = pd.read_csv(args.participants_file)

    #load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CS_CNN(opt).to(device)
    checkpoint = torch.load(args.model_state, map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])

    results = apply_model(opt, model, participants_df, name)

    #save results
    folder_path = os.path.dirname(args.model_state)
    file_name = f"predictions_{name}.csv"
    full_path = os.path.join(folder_path, file_name)
    results.to_csv(full_path, index=False)
    print(f"Saved results to CSV to: {full_path}.") 





