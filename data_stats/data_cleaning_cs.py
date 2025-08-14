"""
This script is used to understand the process of data cleaning and preparation for the thesis project.
"""


import pandas as pd
import numpy as np
import os
import sys
import json
from argparse import Namespace

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
tools_path =os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools'))
sys.path.append(scripts_path)
sys.path.append(tools_path)
from loader_CS import loader3D as loader
from loader_all_fs_cs import loader3D as all_fs_loader


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
def extract_class_at_baseline(participant_id, project_data_dir='/mimer/NOBACKUP/groups/thesis_brainage/data'):
    classification = "CN"
    file_path = os.path.join(project_data_dir, str(participant_id), 'sessions.csv')
    if not os.path.exists(file_path):
        return None
    session_file = pd.read_csv(file_path)
    if session_file.iloc[0]['cognitiveyly_normal'] == False:
        classification = "CI"
    return classification

def extract_class_at_final(participant_id, project_data_dir='/mimer/NOBACKUP/groups/thesis_brainage/data'):
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



if __name__ == "__main__":
    """
    Load the arguments from the json file.
    """
    args1 = load_args_from_json('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/CS_CNN/run_details.json')
    args1.remove_fs = False
    args = load_args_from_json('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/CS_CNN/run_details.json')

    """
    Load the participants file and extract the complete number of participants.
    """
    df = pd.read_csv('/mimer/NOBACKUP/groups/brainage/data/oasis3/participants.tsv', sep='\t')
    num_before_CI_removal = len(df)

    """
    Add classification to the dataframe and remove CI participants.
    """
    df = add_classification(df, args1.project_data_dir)
    df = df[(df['class_at_baseline'] == 'CN') & (df['class_at_final'] == 'CN')] # Exclude CI participants
    num_after_CI_removal = len(df)

    participants = df[['participant_id', 'sex']]
    df_all_fs = all_fs_loader(args1, participants)

    """
    Split by sex
    """
    female_participants = df[df['sex']== 'F']
    male_participants = df[df['sex']== 'M']
    num_of_clean_F_participants = len(female_participants)
    num_of_clean_M_participants = len(male_participants)

    df_M = male_participants[['participant_id', 'sex']]
    df_F = female_participants[['participant_id', 'sex']]


    clean_participants = pd.concat([df_M, df_F], ignore_index=True)
    
    df_M = loader(args, df_M).demo
    df_F = loader(args, df_F).demo
    df_all = loader(args, clean_participants).demo
    min = df_all['age'].min()
    max = df_all['age'].max()
    mean = df_all['age'].mean()


    """
    Print the results of the cleaning process.
    """
    print(f"Number of participants before CI removal: {num_before_CI_removal}")
    print(f"Number of participants after CI removal: {num_after_CI_removal}")
    print(f"Number of female clean participants: {num_of_clean_F_participants}")
    print(f"Number of male clean participants: {num_of_clean_M_participants}")
    print(f"Filtering field strenght results in {len(df_M)} male data and {len(df_F)} female data. Total : {len(df_M) + len(df_F)} data.")
    print(f"Minimum age: {min}, Maximum age: {max}, Mean age: {mean}")
