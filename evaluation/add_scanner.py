import os
import numpy as np
import pandas as pd
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--predictions_file', default='blank', type=str, help="directory of the results to be evaluated")

    args = parser.parse_args()

    return args

def add_scanner_for_path(path):
    json_string = '.json'
    for file in os.listdir(path):
        if json_string in file:
            path = os.path.join(path, file)
            with open(path, 'r') as f:
                data = json.load(f)
                manufacturer = data.get('Manufacturer')
                model_name = data.get('ManufacturersModelName')
                scanner = f"{manufacturer}_{model_name}"
                return scanner

    # If no .json file is found
    print(f"⚠️ No JSON file found in: {path}")
    return "UNKNOWN_UNKNOWN"

def add_scanners_for_datapoint(participant_id, session_id1, session_id2, data_dir = '/mimer/NOBACKUP/groups/brainage/data/oasis3'):
    scan_folder1 = os.path.join(data_dir, participant_id, session_id1, 'anat')
    scan_folder2 = os.path.join(data_dir, participant_id, session_id2, 'anat')
    scanner1 = add_scanner_for_path(scan_folder1)
    scanner2 = add_scanner_for_path(scan_folder2)

    return scanner1, scanner2


if __name__ == "__main__":
    args = parse_args()

    predictions_file = pd.read_csv(args.predictions_file)

    scanner1_list = []
    scanner2_list = []

    for _,row in predictions_file.iterrows():
        participant_id = row['Participant_ID']
        session_id1 = row['Session 1']
        session_id2 = row['Session 2']
        scanner1, scanner2 = add_scanners_for_datapoint(participant_id, session_id1, session_id2)
        scanner1_list.append(scanner1)
        scanner2_list.append(scanner2)

    predictions_file['Scanner 1'] = scanner1_list
    predictions_file['Scanner 2'] = scanner2_list
    
    save_path = os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/further_analysis_results',f"{args.predictions_file}_ws.csv")
    predictions_file.to_csv(save_path)

        
