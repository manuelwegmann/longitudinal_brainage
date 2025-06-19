import os
import numpy as np
import pandas as pd

participants_file = pd.read_csv('/mimer/NOBACKUP/groups/brainage/data/oasis3/participants.tsv', sep="\t")

parent_dir = "/mimer/NOBACKUP/groups/brainage/data/oasis3"
search_string = "sub-OAS"

participant_ids = []
session_ids = []
field_strengths = []
weak_participants = []
weak_ids = []

# Iterate over all items in the directory
for participant_folder in os.listdir(parent_dir):
    participant_path = os.path.join(parent_dir, participant_folder)

    if os.path.isdir(participant_path) and search_string in participant_folder:

        session_string = "ses"

        for session_folder in os.listdir(participant_path):
            session_path = os.path.join(participant_path, session_folder)

            if os.path.isdir(session_path) and session_string in session_folder:

                scans_file_path = os.path.join(session_path, "scans.tsv")
                if not os.path.exists(scans_file_path):
                    print(f"Warning: 'scans.tsv' file not found in {session_path} for participant {participant_folder}. Skipping this session.")
                    continue

                scans_file = pd.read_csv(scans_file_path, sep="\t")
                field_strength = scans_file['fieldstrength'].iloc[0]
                if field_strength < 2:
                    weak_participants.append(participant_folder)
                    weak_ids.append(session_folder)

                participant_ids.append(participant_folder)
                session_ids.append(session_folder)
                field_strengths.append(field_strength)

participant_ids = np.array(participant_ids)
weak_participants = np.array(weak_participants)
weak_ids = np.array(weak_ids)
session_ids = np.array(session_ids)
field_strengths = np.array(field_strengths)

unique, counts = np.unique(field_strengths, return_counts=True)

for val, count in zip(unique, counts):
    print(f"Value: {val}, Count: {count}")

print(f"weak participants:")
unique, counts = np.unique(weak_participants, return_counts=True)
total_count = 0
for val, count in zip(unique, counts):
    overview = participants_file[participants_file['participant_id'] == val]
    mr_sessions = overview['mr_sessions'].values[0]
    if not overview.empty and mr_sessions > 1:
        print(f"Participant: {val}, Count: {count}, MRI sessions: {mr_sessions}")
        total_count += count
    else:
        if overview.empty:
            print(f"There is something wrong with the participants file for {val}.")
        
# Create a DataFrame with weak_participants and their corresponding session_ids
weak_df = pd.DataFrame({
    'weak_participants': weak_participants,
    'weak_ids': weak_ids
})

print(f"Total number of weak scans for patients with multiple sessions: {total_count}")

# Save the DataFrame to a CSV file
weak_df.to_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/further_analysis_results','weak_participants.csv'), index=False)
