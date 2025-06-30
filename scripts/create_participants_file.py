"""
This script creates a .csv file that contains all participant IDs with respective sex and age at baseline.
"""

import pandas as pd
import os
import numpy as np



# For a given participant ID, find CN/CI classification at baseline and final session
def extract_class_at_baseline(participant_id, folder_path='/mimer/NOBACKUP/groups/brainage/data/oasis3'):
    classification = "CN"
    file_path = os.path.join(folder_path, str(participant_id), 'sessions.tsv')
    if pd.read_csv(file_path, sep='\t').iloc[0]['cognitiveyly_normal'] == False:
        classification = "CI"
    return classification

def extract_class_at_final(participant_id, folder_path='/mimer/NOBACKUP/groups/brainage/data/oasis3'):
    classification = "CN"
    file_path = os.path.join(folder_path, str(participant_id), 'sessions.tsv')
    session_file = pd.read_csv(file_path, sep='\t')
    if session_file.iloc[session_file.shape[0] - 1]['cognitiveyly_normal'] == False:
        classification = "CI"
    return classification

# Update the whole dataset with correct classifications at baseline and final session
def add_classification(df, folder_path='/mimer/NOBACKUP/groups/brainage/data/oasis3'):
    df['class_at_baseline'] = df['participant_id'].apply(lambda participant_id: extract_class_at_baseline(participant_id, folder_path))
    df['class_at_final'] = df['participant_id'].apply(lambda participant_id: extract_class_at_final(participant_id, folder_path))
    return df



#check if there are any participants with only 1 scan
def exclude_single_scan_participants(df):
    filtered_df = df[df['mr_sessions'] >= 2]
    excluded_df = df[df['mr_sessions'] < 2]
    return filtered_df

#check if there are any participants with CI at baseline or final session
def exclude_CI_participants(df):
    filtered_df = df[(df['class_at_baseline'] == 'CN') & (df['class_at_final'] == 'CN')]
    return filtered_df



#check the whole dataset if any folders with data are missing
def check_folders_exist(df, folder_path = '/mimer/NOBACKUP/groups/brainage/data/oasis3'):
    checked_df = df
    for participant_id in checked_df['participant_id']:
        if os.path.exists(os.path.join(folder_path, str(participant_id))) == False:
            print(f"Warning: Folder for participant {participant_id} does not exist and will be deleted.")
            index_to_drop = df[df['participant_id'] == participant_id].index
            checked_df = checked_df.drop(index_to_drop)
    return checked_df



def load_participants(project_data_dir = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/data', folder_path = '/mimer/NOBACKUP/groups/brainage/data/oasis3', add_age = False):
    """
    Input:
        folder_path: path to the folder containing the participants.tsv file
        add_age: whether to add age
    Output:
        df: dataframe with the participants and their gender (possibly age)
    """
    participants_file_path = os.path.join(folder_path, 'participants.tsv')
    df = pd.read_csv(participants_file_path, sep='\t')
    df = check_folders_exist(df, folder_path) #delete participants that do not have a folder
    df = add_classification(df, folder_path) #add classification to the dataframe
    df = exclude_CI_participants(df)
    df = exclude_single_scan_participants(df)
    if add_age:
        filtered_rows = []
        for _, row in df.iterrows():
            participant_id = str(row['participant_id'])
            sessions_file_path = os.path.join(project_data_dir, participant_id, 'sessions.csv')

            if os.path.exists(sessions_file_path):
                sessions_file = pd.read_csv(sessions_file_path)
                age_values = sessions_file['age'].dropna()
                if not age_values.empty:
                    row['age'] = age_values.iloc[0]
                    filtered_rows.append(row)
                else:
                    print(f"No age values at all for {participant_id}.")

        df = pd.DataFrame(filtered_rows).reset_index(drop=True)
        return df[['participant_id', 'sex', 'age']]

    else:
        return df[['participant_id', 'sex']]
    

if __name__ == "__main__":
    df_wo_age = load_participants()
    df_w_age = load_participants(add_age = True)
    df_wo_age.to_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/further_analysis_results/participants_file_without_age.csv')
    df_w_age.to_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/further_analysis_results/participants_file_with_age.csv')