import numpy as np
import pandas as pd
import os


#function to load standard dataset description
def load_basic_overview(file_path): #filepath points to participants.tsv file in OASIS-3 folder
    df = pd.read_csv(file_path, sep='\t')
    return df


#for a given participant ID, find their actual age at baseline
def extract_age_at_baseline(participant_id, folder_path = '/mimer/NOBACKUP/groups/brainage/data/oasis3'): #extract correct age at baseline
    file_path = os.path.join(folder_path, str(participant_id), 'sessions.tsv')  
    return pd.read_csv(file_path, sep='\t').iloc[0]['age']


#update whole dataset with corect ages at baseline
def add_ages(df, folder_path):  # df is the DataFrame, and folder_path points to the OASIS-3 folder
    df['age'] = df['participant_id'].apply(extract_age_at_baseline)
    return df


def extract_class_at_baseline(participant_id, folder_path = '/mimer/NOBACKUP/groups/brainage/data/oasis3'):
    classification = "CN"
    file_path = os.path.join(folder_path, str(participant_id), 'sessions.tsv')
    if pd.read_csv(file_path, sep='\t').iloc[0]['cognitiveyly_normal'] == False:
        classification = "CI"
    return classification

def extract_class_at_final(participant_id, folder_path = '/mimer/NOBACKUP/groups/brainage/data/oasis3'):
    classification = "CN"
    file_path = os.path.join(folder_path, str(participant_id), 'sessions.tsv')
    session_file = pd.read_csv(file_path, sep='\t')
    if session_file.iloc[session_file.shape[0]-1]['cognitiveyly_normal'] == False:
        classification = "CI"
    return classification
    

def add_classification(df, folder_path = '/mimer/NOBACKUP/groups/brainage/data/oasis3'):
    df['class_at_baseline'] = df['participant_id'].apply(extract_class_at_baseline)
    df['class_at_final'] = df['participant_id'].apply(extract_class_at_final)
    return df


#extract time between first and last scan for a given participant id
def extract_duration(participant_id, folder_path = '/mimer/NOBACKUP/groups/brainage/data/oasis3'):
    file_path = os.path.join(folder_path, str(participant_id), 'sessions.tsv')  
    sessions_file = pd.read_csv(file_path, sep='\t')
    num_sessions = sessions_file.shape[0]
    baseline = sessions_file.iloc[0]['days_from_baseline']
    final = sessions_file.iloc[num_sessions-1]['days_from_baseline']
    return final - baseline

def add_duration(df):
    df['duration'] = df['participant_id'].apply(extract_duration)
    return df


#check if for a given participant_id there exists a folder with scans
def check_folder_exists(participant_id, folder_path): #folder_path points to the OASIS-3 folder
    path_to_subject_folder = os.path.join(folder_path, str(participant_id))
    return os.path.exists(path_to_subject_folder)


#check the whole dataset if any folders with data are missing
def check_folders_exist(df, folder_path, delete_rows = True):
    checked_df = df
    for participant_id in checked_df['participant_id']:
        if check_folder_exists(participant_id,folder_path)==False:
            print(f"Warning: Folder for participant {participant_id} does not exist.")
            if delete_rows == True:
                index_to_drop = df[df['participant_id'] == participant_id].index
                checked_df = checked_df.drop(index_to_drop)
                print(f"Participant {participant_id} was deleted from the dataframe.")

    return checked_df

def exclude_single_scan_participants(df):
    filtered_df = df[df['mr_sessions'] >= 2]
    excluded_df = df[df['mr_sessions'] < 2]
    print(f'There are {filtered_df.shape[0]} subjects with at least 2 scans.')
    print(f'There are {excluded_df.shape[0]} subjects with only 1 scan.')
    return filtered_df

def exclude_by_duration(df, duration_threshold = 30):
    filtered_df = df[df['duration'] >= duration_threshold]
    return filtered_df


#split dataset into female and male
def split_by_gender(df):
    df_male = df[df['sex'] == 'M']
    df_female = df[df['sex'] == 'F']
    return df_male, df_female

def split_by_class(df):
    df_CN = df[df['class_at_baseline'] == "CN"]
    df_CI = df[df['class_at_baseline'] == "CI"]
    return df_CN, df_CI

def full_data_load(fp_participants = '/mimer/NOBACKUP/groups/brainage/data/oasis3/participants.tsv', fp_oasis = '/mimer/NOBACKUP/groups/brainage/data/oasis3'):
    df = load_basic_overview(file_path = fp_participants)
    df = check_folders_exist(df=df, folder_path=fp_oasis)
    df = add_ages(df=df, folder_path=fp_oasis)
    df = add_duration(df)
    df = add_classification(df)
    print(df.head())
    return df
