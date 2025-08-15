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
from loader import loader3D
from loader_all_fs import all_fs_loader3D


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
    args = load_args_from_json('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC/run_details.json')

    """
    Load the participants file and extract the complete number of participants.
    """
    df = pd.read_csv('/mimer/NOBACKUP/groups/brainage/data/oasis3/participants.tsv', sep='\t')
    num_before_CI_removal = len(df)

    """
    Add classification to the dataframe and remove CI participants.
    """
    df = add_classification(df, args.project_data_dir)
    df_switch = df[(df['class_at_baseline'] == 'CN') & (df['class_at_final'] == 'CI')]
    print(f"Number of participants with switch from CN to CI: {len(df_switch)}")
    df = df[(df['class_at_baseline'] == 'CN') & (df['class_at_final'] == 'CN')] # Exclude CI participants
    num_before_single_scan_removal = len(df)

    """
    Remove single-scan participants and split by sex. Extract participants with 3 or more scans.
    """
    df = df[df['mr_sessions'] > 1]  # Exclude participants with only one scan
    num_after_single_scan_removal = len(df)
    female_participants = df[df['sex']== 'F']
    male_participants = df[df['sex']== 'M']
    num_of_clean_F_participants = len(female_participants)
    num_of_clean_M_participants = len(male_participants)
    many_scans_male = male_participants[male_participants['mr_sessions'] > 2]
    many_scans_female = female_participants[female_participants['mr_sessions'] > 2]
    df_U = df[df['sex']=='U']

    df_M = male_participants[['participant_id', 'sex']]
    df_F = female_participants[['participant_id', 'sex']]


    clean_participants = pd.concat([df_M, df_F], ignore_index=True)
    clean_participants.to_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/archive/further_analysis_results/cp.csv', index=False)

    df_all_fs_M = all_fs_loader3D(args, df_M)
    df_all_fs_F = all_fs_loader3D(args, df_F)
    df_M = loader3D(args, df_M)
    df_F = loader3D(args, df_F)

    """
    Print the results of the cleaning process.
    """
    print(f"Number of participants before CI removal: {num_before_CI_removal}")
    print(f"Number of participants before single scan removal: {num_before_single_scan_removal}")
    print(f"Number of participants after single scan removal: {num_after_single_scan_removal}")
    print(f"Number of female clean participants: {num_of_clean_F_participants}")
    print(f"Number of male clean participants: {num_of_clean_M_participants}")
    print(f"Total number of participants with more than 2 scans: {len(many_scans_female)+len(many_scans_male)}")
    print(f"Numbers of participants with no sex: {len(df_U)}. Note that none of them have age information at any timepoint.")
    print(f"No filtering field strenght results in {len(df_all_fs_M)} male pairs and {len(df_all_fs_F)} female pairs. Total: {len(df_all_fs_M) + len(df_all_fs_F)} pairs.")
    print(f"Filtering field strenght results in {len(df_M)} male pairs and {len(df_F)} female pairs. Total : {len(df_M) + len(df_F)} pairs.")

    max_scans_M = many_scans_male['mr_sessions'].max()
    max_scans_F = many_scans_female['mr_sessions'].max()
    print(f"Maximum number of scans for one participant: {max(max_scans_M, max_scans_F)}")

    """
    Check how many clean participants have multiple field strengths.
    """
    df = all_fs_loader3D(args, clean_participants).demo
    ids = []
    sessions = []
    field_strengths = []
    for _, row in df.iterrows():
        id = row['participant_id']
        if id in ids:
            continue
        part_ids = []
        part_sessions = []
        part_field_strengths = []
        sub_df = df[df['participant_id'] == id]
        for _, sub_row in sub_df.iterrows():
            if sub_row['session_id1'] not in part_sessions:
                part_ids.append(id)
                part_sessions.append(sub_row['session_id1'])
                part_field_strengths.append(sub_row['field_strength1']) 
            if sub_row['session_id2'] not in part_sessions:
                part_ids.append(id)
                part_sessions.append(sub_row['session_id2'])
                part_field_strengths.append(sub_row['field_strength2'])
        ids.extend(part_ids)
        sessions.extend(part_sessions)
        field_strengths.extend(part_field_strengths)

    print(f"type(ids): {type(ids)}, length: {len(ids)}")
    print(f"type(sessions): {type(sessions)}, length: {len(sessions)}")
    print(f"type(field_strengths): {type(field_strengths)}, length: {len(field_strengths)}")
    if len(ids) > 0:
        print(f"First element in ids: {type(ids[0])}")
    if len(sessions) > 0:
        print(f"First element in sessions: {type(sessions[0])}")
    if len(field_strengths) > 0:
        print(f"First element in field_strengths: {type(field_strengths[0])}")

    clean_participants_with_ses_fs = pd.DataFrame({'participant_id': ids, 'session_id': sessions, 'field_strength': field_strengths})
    count = 0
    ids = []
    for _, row in clean_participants_with_ses_fs.iterrows():
        id = row['participant_id']
        if id in ids:
            continue
        sub_df = clean_participants_with_ses_fs[clean_participants_with_ses_fs['participant_id'] == id]
        num_unique_fs = sub_df['field_strength'].nunique()   
        if num_unique_fs > 1:
            ids.append(id)
            count += 1
    print(f"Number of clean participants with multiple field strengths: {count}")