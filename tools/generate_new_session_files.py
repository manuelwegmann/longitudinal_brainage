import os
import pandas as pd
import numpy as np

def generate_new_sessions_file(participant_id, original_data_dir = '/mimer/NOBACKUP/groups/brainage/data/oasis3', project_data_dir = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/data'):
    """
    Generates a new sessions.tsv file for a participant in the specified project data directory.
    
    Args:
        participant_id (str): The ID of the participant.
        original_data_dir (str): The path to the original data directory.
        project_data_dir (str): The path to the project data directory.
    """
    sessions_file_path = os.path.join(original_data_dir, participant_id, "sessions.tsv")

    if not os.path.exists(sessions_file_path):
        print(f"Warning: 'sessions.tsv' file not found for participant {participant_id} in original datadirectory. Skipping this participant.")
        return None
        
    sessions_file = pd.read_csv(sessions_file_path, sep="\t")
    age = sessions_file['age'].values
    dfb = sessions_file['days_from_baseline'].values
    dfd = sessions_file['days_to_diagnosis'].values
    non_nan_mask_age = ~np.isnan(age)
    if not np.any(non_nan_mask_age):
        print(f"There is no age entry at all for participant {participant_id}.")
        return 1
    else:
        return 0


if __name__ == "__main__":
    participants_df = pd.read_csv('/mimer/NOBACKUP/groups/brainage/data/oasis3/participants.tsv', sep="\t")
    project_dir = "/mimer/NOBACKUP/groups/brainage/thesis_brainage"
    for i in range(len(participants_df)):
        participant_id = participants_df['participant_id'].iloc[i]
        new_folder_path = os.path.join(project_dir, "data", participant_id)
        os.makedirs(new_folder_path, exist_ok=True)

    valid_rows = []
    for index, row in participants_df.iterrows():
        participant_id = row['participant_id']
        a = generate_new_sessions_file(participant_id)
        if a is not None:
            if a == 1:
                valid_rows.append(row)

    a = pd.DataFrame(valid_rows).reset_index(drop=True)
    print(a.head())

    unique, counts = np.unique(a['mr_sessions'].values, return_counts=True)
    print(unique)
    print(counts)

    
    


