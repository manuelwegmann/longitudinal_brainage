import pandas as pd
import os
import numpy as np

from prep_data import add_classification, check_folders_exist, exclude_single_scan_participants

def load_participants(folder_path = '/mimer/NOBACKUP/groups/brainage/data/oasis3', add_age = False):
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
    df = exclude_single_scan_participants(df)

    df = df[(df['class_at_baseline'] == 'CI') | (df['class_at_final'] == 'CI')]

    if add_age:
        filtered_rows = []
        for _, row in df.iterrows():
            participant_id = str(row['participant_id'])
            sessions_file_path = os.path.join(folder_path, participant_id, 'sessions.tsv')

            if os.path.exists(sessions_file_path):
                sessions_file = pd.read_csv(sessions_file_path, sep='\t')
                age_values = sessions_file['age'].dropna()
                if not age_values.empty:
                    row['age'] = age_values.iloc[0]
                    filtered_rows.append(row)

        df = pd.DataFrame(filtered_rows).reset_index(drop=True)
        return df[['participant_id', 'sex', 'age']]

    else:
        return df[['participant_id', 'sex']]
    
CI_participants_wo_age = load_participants(add_age=False)

save_path_wo_age = os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/CI_participants', 'CI_participants.csv')

CI_participants_wo_age.to_csv(save_path_wo_age, index=False)

print(f"Saved CI participants without age to {save_path_wo_age}.")


