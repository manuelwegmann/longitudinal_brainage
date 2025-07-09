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

    not_nan_mask = df['age'].notna()
    if not_nan_mask.any():
        first_valid_index = df[not_nan_mask].index[0]
        valid_age = df.loc[first_valid_index, 'age']
        valid_days = df.loc[first_valid_index, 'days_from_baseline']
        for i in range(num_sessions):
            if np.isnan(df['age'].iloc[i]):
                df.loc[i, 'age'] = valid_age + (df.loc[i, 'days_from_baseline'] - valid_days) / 365
        return df
    else:
        # If all ages are NaN, we cannot calculate a valid age, so we return the original df
        print(f"Warning: All ages are NaN for participant {participant_id}. Returning original df.")
        return df


#30422,30486,30754



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


    
    


