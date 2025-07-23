"""
This script creates to files containing participant ids that can be used for training longitudinal and cross-sectional models.
"""


import pandas as pd
import numpy as np
import os

import json
from argparse import Namespace


"""
Function to load arguments from a json file to work with the data loader.
"""
def load_args_from_json(filepath):

    with open(filepath, 'r') as f:
        args_dict = json.load(f)

    args = Namespace(**args_dict)

    return args

"""
These functions are to classify participants based on their cognitive status.
"""
# For a given participant ID, find CN/CI classification at baseline and final session
def extract_class_at_baseline(participant_id, project_data_dir='/mimer/NOBACKUP/groups/brainage/thesis_brainage/data'):
    classification = "CN"
    file_path = os.path.join(project_data_dir, str(participant_id), 'sessions.csv')
    if not os.path.exists(file_path):
        return None
    session_file = pd.read_csv(file_path)
    if session_file.iloc[0]['cognitiveyly_normal'] == False:
        classification = "CI"
    return classification

def extract_class_at_final(participant_id, project_data_dir='/mimer/NOBACKUP/groups/brainage/thesis_brainage/data'):
    classification = "CN"
    file_path = os.path.join(project_data_dir, str(participant_id), 'sessions.csv')
    if not os.path.exists(file_path):
        return None
    session_file = pd.read_csv(file_path)
    if session_file.iloc[session_file.shape[0] - 1]['cognitiveyly_normal'] == False:
        classification = "CI"
    return classification

# Update the whole dataset with correct classifications at baseline and final session
def add_classification(df, project_data_dir='/mimer/NOBACKUP/groups/brainage/data/oasis3'):
    df['class_at_baseline'] = df['participant_id'].apply(lambda participant_id: extract_class_at_baseline(participant_id, project_data_dir))
    df['class_at_final'] = df['participant_id'].apply(lambda participant_id: extract_class_at_final(participant_id, project_data_dir))
    return df

#check the whole dataset if any folders with data are missing
def check_folders_exist(df, folder_path = '/mimer/NOBACKUP/groups/brainage/data/oasis3'):
    checked_df = df
    for participant_id in checked_df['participant_id']:
        if os.path.exists(os.path.join(folder_path, str(participant_id))) == False:
            print(f"Warning: Folder for participant {participant_id} does not exist and will be deleted.")
            index_to_drop = df[df['participant_id'] == participant_id].index
            checked_df = checked_df.drop(index_to_drop)
    return checked_df



if __name__ == "__main__":
    """
    Load the participants file and extract the complete number of participants.
    """
    df = pd.read_csv('/mimer/NOBACKUP/groups/brainage/data/oasis3/participants.tsv', sep='\t')
    df = check_folders_exist(df)
    print(f"Total number of participants: {len(df)}")

    """
    Add classification to the dataframe and remove CI participants.
    """
    df = add_classification(df, '/mimer/NOBACKUP/groups/brainage/thesis_brainage/data')
    df = df[(df['class_at_baseline'] == 'CN') & (df['class_at_final'] == 'CN')]
    print(f"Number of participants after CI removal: {len(df)}. These are the participants suitable for training cross-sectional models.")
    
    """
    Add ages to the dataframe.
    """
    valid_rows = []
    ages = []
    for _, row in df.iterrows():
        participant_id = row['participant_id']
        file_path = os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/data', str(participant_id), 'sessions.csv')
        session_file = pd.read_csv(file_path)
        if session_file.empty:
            continue
        age = session_file.iloc[0]['age']
        if pd.isna(age):
            continue
        valid_rows.append(row)
        ages.append(age)
    df = pd.DataFrame(valid_rows)
    df['age'] = ages
    print(f"Number of participants with valid ages: {len(df)}")
    participants_cs = df[['participant_id','sex','age']]
    participants_cs.to_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/participants_cs.csv', index=False)
    df = df[df['mr_sessions']>= 2]
    participants = df[['participant_id','sex','age']]
    participants.to_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/participants.csv', index=False)
    print(f"Number of participants with at least 2 sessions: {len(df)}")