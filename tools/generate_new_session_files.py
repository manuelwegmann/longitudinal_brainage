import os
import pandas as pd
import numpy as np

def generate_new_sessions_df(participant_id, original_data_dir = '/mimer/NOBACKUP/groups/brainage/data/oasis3', project_data_dir = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/data'):
    """
    Generates a new sessions df for a participant in the specified project data directory.
    
    Args:
        participant_id (str): The ID of the participant.
        original_data_dir (str): The path to the original data directory.
        project_data_dir (str): The path to the project data directory.
    """
    sessions_file_path = os.path.join(original_data_dir, participant_id, "sessions.tsv")

    if not os.path.exists(sessions_file_path):
        print(f"Warning: 'sessions.tsv' file not found for participant {participant_id} in original datadirectory. Skipping this participant.")
        return None
        
    df = pd.read_csv(sessions_file_path, sep = '\t')

    num_sessions = len(df)
    baseline_age = df['age'].iloc[0]
    baseline_days = df['days_from_baseline'].iloc[0]

    if np.isnan(baseline_age):
        return df
    else:
        for i in range(num_sessions):
            if np.isnan(df['age'].iloc[i]):
                df.loc[i, 'age'] = baseline_age + (df.loc[i, 'days_from_baseline'] - baseline_days) / 365
        return df


if __name__ == "__main__":
    participants_df = pd.read_csv('/mimer/NOBACKUP/groups/brainage/data/oasis3/participants.tsv', sep="\t")
    project_dir = "/mimer/NOBACKUP/groups/brainage/thesis_brainage"

    #generate new folders in project directory to save new session files.
    for i in range(len(participants_df)):
        participant_id = participants_df['participant_id'].iloc[i]
        new_folder_path = os.path.join(project_dir, "data", participant_id)
        os.makedirs(new_folder_path, exist_ok=True)

    #generate and save new session files.
    for index, row in participants_df.iterrows():
        participant_id = row['participant_id']
        new_sessions_df = generate_new_sessions_df(participant_id)
        if new_sessions_df is not None:
            new_sessions_df.to_csv(os.path.join(project_dir,'data',str(participant_id),'sessions.csv'), index=False)


    
    


