import pandas as pd
import os
import numpy as np

from prep_data import add_classification, check_folders_exist, exclude_single_scan_participants

def load_CI_participants(folder_path = '/mimer/NOBACKUP/groups/brainage/data/oasis3', project_data_dir = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/data', bothCI = True):

    participants_file_path = os.path.join(folder_path, 'participants.tsv')
    df = pd.read_csv(participants_file_path, sep='\t')
    df = check_folders_exist(df, folder_path) #delete participants that do not have a folder
    df = add_classification(df, folder_path) #add classification to the dataframe
    df = exclude_single_scan_participants(df)

    if bothCI:
        df = df[(df['class_at_baseline'] == 'CI') & (df['class_at_final'] == 'CI')]
    if not bothCI:
        df = df[(df['class_at_baseline'] == 'CI') | (df['class_at_final'] == 'CI')]

    #extract ages at baseline
    filtered_rows = []
    for _, row in df.iterrows():
        participant_id = str(row['participant_id'])
        sessions_file_path = os.path.join(project_data_dir, participant_id, 'sessions.csv')
        if os.path.exists(sessions_file_path):
            sessions_file = pd.read_csv(sessions_file_path, sep=',')
            age_values = sessions_file['age'].dropna()
            if not age_values.empty:
                row['age'] = age_values.iloc[0]
                filtered_rows.append(row)
            else:
                print(f"Warning: No age found for participant {participant_id}. Skipping this participant.")
        else:
            print(f"Warning: 'sessions.tsv' file not found for participant {participant_id}. Skipping this participant.")  
    df = pd.DataFrame(filtered_rows).reset_index(drop=True)

    duration = []

    for _, row in df.iterrows():
        participant_id = str(row['participant_id'])
        sessions_file_path = os.path.join(project_data_dir, participant_id, 'sessions.csv')
        sessions_file = pd.read_csv(sessions_file_path, sep=',')
        time1 = sessions_file['days_from_baseline'].min()
        time2 = sessions_file['days_from_baseline'].max()
        duration.append((time2 - time1)/365)

    df['duration'] = duration
        
    return df


def load_CN_participants(folder_path = '/mimer/NOBACKUP/groups/brainage/data/oasis3', project_data_dir = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/data'):

    participants_file_path = os.path.join(folder_path, 'participants.tsv')
    df = pd.read_csv(participants_file_path, sep='\t')
    df = check_folders_exist(df, folder_path) #delete participants that do not have a folder
    df = add_classification(df, folder_path) #add classification to the dataframe
    df = exclude_single_scan_participants(df)

    df = df[(df['class_at_baseline'] == 'CN') & (df['class_at_final'] == 'CN')]

    #extract ages at baseline
    filtered_rows = []
    for _, row in df.iterrows():
        participant_id = str(row['participant_id'])
        sessions_file_path = os.path.join(project_data_dir, participant_id, 'sessions.csv')
        if os.path.exists(sessions_file_path):
            sessions_file = pd.read_csv(sessions_file_path, sep=',')
            age_values = sessions_file['age'].dropna()
            if not age_values.empty:
                row['age'] = age_values.iloc[0]
                filtered_rows.append(row)
            else:
                print(f"Warning: No age found for participant {participant_id}. Skipping this participant.")
        else:
            print(f"Warning: 'sessions.tsv' file not found for participant {participant_id}. Skipping this participant.")  
    df = pd.DataFrame(filtered_rows).reset_index(drop=True)

    duration = []

    for _, row in df.iterrows():
        participant_id = str(row['participant_id'])
        sessions_file_path = os.path.join(project_data_dir, participant_id, 'sessions.csv')
        sessions_file = pd.read_csv(sessions_file_path, sep=',')
        time1 = sessions_file['days_from_baseline'].min()
        time2 = sessions_file['days_from_baseline'].max()
        duration.append((time2 - time1)/365)

    df['duration'] = duration
        
    return df

def find_closest_match(age, duration, df):
    error = np.inf
    closest_row = None
    for _, row in df.iterrows():
        error_new = (age-row['age'])**2 + (duration-row['duration'])**2
        if error_new < error:
            error = error_new
            closest_row = row.copy()
            closest_row['error'] = error_new
    return pd.DataFrame([closest_row])
    

if __name__ == "__main__":
    
    CI_participants = load_CI_participants()
    print(len(CI_participants))
    print(CI_participants.head())
    CN_participants = load_CN_participants()

    CN_participants = CN_participants[CN_participants['mr_sessions'] <= 3]

    closest_rows = []
    for _, row in CI_participants.iterrows():
        closest_row = find_closest_match(row['age'], row['duration'], CN_participants)
        closest_rows.append(closest_row)


    closest_df = pd.concat(closest_rows, ignore_index=True)
    print(closest_df.head())
    print(len(closest_df))
    print(f"max error: {closest_df['error'].max()}, min error: {closest_df['error'].min()}, mean error: {closest_df['error'].mean()}")

    CI_participants.to_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/participant_files/CI_participants.csv', index=False)
    closest_df.to_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/participant_files/CN_controlgroup.csv', index=False)