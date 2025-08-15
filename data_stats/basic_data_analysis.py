"""
Script to perform basic data analysis on the OASIS-3 dataset.
"""

import numpy as np
import os
import pandas as pd
import json

from prep_data import add_classification, check_folders_exist, add_ages, add_duration



if __name__ == "__main__":
    particicipants_filepath = '/mimer/NOBACKUP/groups/brainage/data/oasis3/participants.tsv'
    df = pd.read_csv(particicipants_filepath, sep='\t')

    df_M = df[df['sex']== 'M']
    df_F = df[df['sex']== 'F']
    df_U = df[df['sex']== 'U']
    print(f"Number of participants in OASIS-3 dataset (M,F,U): {len(df)} ({len(df_M)}, {len(df_F)}, {len(df_U)})")
    print(f"Number of scans: {np.sum(df['mr_sessions'])}")
    print(f"Race: {df['race'].value_counts()}")

    df = check_folders_exist(df)
    df = add_classification(df)
    df = add_ages(df)
    print(df.head())
    
    df_CN = df[df['class_at_baseline'] == 'CN']
    df_CI = df[df['class_at_baseline'] == 'CI']
    print(f"Number of participants in OASIS-3 dataset with CN/CI at baseline: {len(df_CN)}, {len(df_CI)}")
    print("CN:", df_CN['sex'].value_counts())
    print("CI:", df_CI['sex'].value_counts())

    print(f"Mean age CN: {df_CN['age'].mean(), df_CN['age'].std()}")
    print(f"Mean age CI: {df_CI['age'].mean(), df_CI['age'].std()}")
    print(f"Mean age total: {df['age'].mean(), df['age'].std()}")

    print(f"Mean education CN: {df_CN['education'].mean(), df_CN['education'].std()}")
    print(f"Mean education CI: {df_CI['education'].mean(), df_CI['education'].std()}")
    print(f"Mean education total: {df['education'].mean(), df['education'].std()}")

    df_long = df[df['mr_sessions'] > 1]
    df_long = add_duration(df_long)
    df_long_CN = df_long[df_long['class_at_baseline'] == 'CN']
    df_long_CI = df_long[df_long['class_at_baseline'] == 'CI']
    print(f"Mean duration CN: {df_long_CN['duration'].mean()}")
    print(f"Mean duration CI: {df_long_CI['duration'].mean()}")
    print(f"Mean duration total: {df_long['duration'].mean()}")

    single_scan = df[df['mr_sessions'] == 1]
    single_scan_CN = single_scan[single_scan['class_at_baseline'] == 'CN']
    single_scan_CI = single_scan[single_scan['class_at_baseline'] == 'CI']
    print(f"Single scan participants: {len(single_scan)}, CN: {len(single_scan_CN)}, CI: {len(single_scan_CI)}")
    print(f"CN number of scans: {df_long_CN['mr_sessions'].value_counts()}")
    print(f"CI number of scans: {df_long_CI['mr_sessions'].value_counts()}")
    print(f"Total number of scans: {df_long['mr_sessions'].value_counts()}")




    parent_dir = "/mimer/NOBACKUP/groups/brainage/data/oasis3"
    search_string = "sub-OAS"

    manufacturer_list = []
    model_list = []

    # Iterate over all items in the directory
    for participant_folder in os.listdir(parent_dir):
        participant_path = os.path.join(parent_dir, participant_folder)

        if os.path.isdir(participant_path) and search_string in participant_folder:

            session_string = "ses"

            for session_folder in os.listdir(participant_path):
                session_path = os.path.join(participant_path, session_folder)

                if os.path.isdir(session_path) and session_string in session_folder:

                    anat_path = os.path.join(session_path, "anat")
                    if not os.path.exists(anat_path):
                        continue
                    
                    des_string = ".json"
                    for file in os.listdir(anat_path):
                        if des_string in file:
                            des_path = os.path.join(anat_path, file)
                            with open(des_path, 'r') as f:
                                data = json.load(f)
                            manufacturer = data.get('Manufacturer')
                            model_name = data.get('ManufacturersModelName')
                            manufacturer_list.append(manufacturer)
                            model_list.append(model_name)
                        else:
                            continue
    
    models = []
    for i in range(len(manufacturer_list)):
        model = f"{manufacturer_list[i]} {model_list[i]}"
        models.append(model)
    
    # Use numpy to get unique models and their counts
    unique_models, counts = np.unique(models, return_counts=True)

    # Print the results
    for model, count in zip(unique_models, counts):
        print(f"{model}: {count}")



    