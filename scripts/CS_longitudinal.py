import pandas as pd
import os
import numpy as np
import argparse
import json

from loader import loader3D

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--predictions_file', default='blank', type=str, help="directory to the cross-sectional prediction file")
    parser.add_argument('--participants_file', default='blank', type=str, help="participants file to be evaluated longitudinally")
    parser.add_argument('--json', default='blank', type=str, help="path to the json file with the arguments")

    args = parser.parse_args()

    return args


def make_longitudinal_prediction_on_single_pair(participant_id, session1, session2, results_df):
    """
    participant_id: participant ID from the custom dataset
    session1: session ID for the first session
    session2: session ID for the second session
    results_df: dataframe containing the results of running the CS model on participants file
    """

    matching_participant_ids = results_df[results_df['Participant_ID'] == participant_id]
    if matching_participant_ids.empty:
        print(f"No results found for participant {participant_id}. Please check for data leakage.")
        return None
    matching_session1_ids = matching_participant_ids[matching_participant_ids['Sessions'] == session1]
    if matching_session1_ids.empty:
        print(f"No results found for session {session1} of participant {participant_id}. Please check for data leakage.")
        return None
    matching_session2_ids = matching_participant_ids[matching_participant_ids['Sessions'] == session2]
    if matching_session2_ids.empty:
        print(f"No results found for session {session2} of participant {participant_id}. Please check for data leakage.")
        return None
    pred1 = matching_session1_ids['Prediction'].values[0]
    pred2 = matching_session2_ids['Prediction'].values[0]
    prediction = pred2 - pred1

    return prediction



if __name__ == "__main__":

    args = parse_args()

    with open(args.json, 'r') as f:
        opt_dict = json.load(f)
        opt = argparse.Namespace(**opt_dict)

    #load the dataset from dataloader
    participants_df = pd.read_csv(args.participants_file)
    lgt_dataset = loader3D(opt, participants_df).demo
    print("Data from dataloader:")
    print(lgt_dataset.head())

    # Load the results from the CS model
    results_CS = pd.read_csv(args.predictions_file)
    print("Results from CS model:")
    print(results_CS.head())

    valid_rows = []
    predictions = []

    for index, row in lgt_dataset.iterrows():
        participant_id = row['participant_id']
        session1 = row['session_id1']
        session2 = row['session_id2']

        prediction = make_longitudinal_prediction_on_single_pair(participant_id, session1, session2, results_CS)
        if prediction is not None:
            valid_rows.append(row)
            predictions.append(prediction)

    
    valid_lgt_dataset = pd.DataFrame(valid_rows).reset_index(drop=True)
    print(valid_lgt_dataset.head())
    predictions = np.array(predictions)

    # Build results DataFrame
    results = pd.DataFrame({
        "Participant_ID": valid_lgt_dataset['participant_id'],
        "Target": valid_lgt_dataset['duration'],
        "Prediction": predictions,
        "Age": valid_lgt_dataset['age'],
        "Sex (M)": valid_lgt_dataset['sex_M'],
        "Session 1":valid_lgt_dataset['session_id1'],
        "Session 2": valid_lgt_dataset['session_id2']
    })

    # Define directory and file path
    csv_path = os.path.join(os.path.dirname(args.predictions_file), 'longitudinal_predictions.csv')
    results.to_csv(csv_path, index=False)
        
