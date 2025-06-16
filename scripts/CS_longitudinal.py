import pandas as pd
import os
import numpy as np
import argparse
import json

from loader import loader3D

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--results_file', default='blank', type=str, help="directory of the results to be evaluated")
    parser.add_argument('--participants_file', default='blank', type=str, help="participants file that lead to results")
    parser.add_argument('--path_to_json', default='blank', type=str, help="path to the json file with the arguments")
    parser.add_argument('--data_dir', default='/mimer/NOBACKUP/groups/brainage/thesis_brainage/data', type=str, help="directory of the data")

    args = parser.parse_args()

    return args


def make_longitudinal_prediction_on_single_pair(participant_id, session1, session2, results_df):

    matching_participant_ids = results_df[results_df['Participant_ID'] == participant_id]
    if matching_participant_ids.empty:
        print(f"No results found for participant {participant_id}.")
        return None
    matching_session1_ids = matching_participant_ids[matching_participant_ids['Sessions'] == session1]
    if matching_session1_ids.empty:
        print(f"No results found for session {session1} of participant {participant_id}.")
        return None
    matching_session2_ids = matching_participant_ids[matching_participant_ids['Sessions'] == session2]
    if matching_session2_ids.empty:
        print(f"No results found for session {session2} of participant {participant_id}.")
        return None
    pred1 = matching_session1_ids['Prediction'].values[0]
    pred2 = matching_session2_ids['Prediction'].values[0]
    prediction = pred2 - pred1

    return prediction



if __name__ == "__main__":

    args = parse_args()

    with open(args.path_to_json, 'r') as f:
        opt_dict = json.load(f)
        opt = argparse.Namespace(**opt_dict)

    participants_df = pd.read_csv(args.participants_file)
    lgt_dataset = loader3D(opt, participants_df).demo
    print(lgt_dataset.head())

    results_CS = pd.read_csv(args.results_file)
    print(results_CS.head())
    check = results_CS[results_CS['Participant_ID'] == 'sub-OAS30284']
    print(check.head())

    valid_rows = []
    predictions = []

    for index, row in lgt_dataset.iterrows():
        participant_id = row['participant_id']
        session1 = row['session_id1']
        session2 = row['session_id2']

        prediction = make_longitudinal_prediction_on_single_pair(participant_id, session1, session2, results_CS)
        if prediction is not None:
            print("WE ARE SO BACK")
            valid_rows.append(row)
            predictions.append(prediction)

    
    valid_lgt_dataset = pd.DataFrame(valid_rows).reset_index(drop=True)
    print(valid_lgt_dataset.head())
    predictions = np.array(predictions)

    # Build results DataFrame
    results = pd.DataFrame({
        "Participant_ID": valid_lgt_dataset['participant_id'],
        "Target": valid_lgt_dataset['age'],
        "Prediction": predictions,
        "Sex (M)": valid_lgt_dataset['sex_M'],
        "Sex (F)": valid_lgt_dataset['sex_F'],
        "Session 1":valid_lgt_dataset['session_id1'],
        "Session 2": valid_lgt_dataset['session_id2']
    })

    # Define directory and file path
    csv_path = os.path.join(os.path.dirname(args.results_file), 'results.csv')
    results.to_csv(csv_path, index=False)
        
