"""
This script contains the data loader for the cross-sectional CNN model.
For each participant that gets passed to the custom dataset, it:
    generates all available scans.
    filters out scans with field strength 1.5 Tesla.
    filters out scans without available age.
"""


from torch.utils.data import Dataset
import pandas as pd
import os
import glob
import torchio as tio
import numpy as np
import torch


def check_fieldstrength(participant_id, session_id, folder_path = '/mimer/NOBACKUP/groups/brainage/data/oasis3'):
    """
    Function to check the field strength of a participant's session.
    Input:
        participant_id: id of the participant
        session_id: id of the session
        folder_path: path to the folder containing all participant folders
    Output:
        field_strength: field strength of the session
    """
    scans_file_path = os.path.join(folder_path, str(participant_id), str(session_id), 'scans.tsv')
    if not os.path.exists(scans_file_path):
        print(f"Warning: 'scans.tsv' file not found for participant {participant_id} in session {session_id}. Skipping this session.")
        return None
    
    else:
        scans_file = pd.read_csv(scans_file_path, sep='\t')
        field_strength = scans_file['fieldstrength'].iloc[0]
        return field_strength

    


def build_participant_block(participant_id, sex, folder_path='/mimer/NOBACKUP/groups/brainage/data/oasis3', project_data_dir = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/data'):
    """
    Function to extract all available scans of a participant. Encode gender as one-hot encoding.
    Only include participants with 3 Tesla scans.
    Input:
        participant_id: id of the participant
        sex: gender of the participant
        folder_path: path to the folder containing all participant folders
        project_data_dir: path to the new data directory containing the updated session files.
    Output:
        Dataframe where each all the scans from the same participant is extracted.
    """

    #one-hot encoding
    sex_M = 0
    sex_F = 0
    sex_U = 0
    if sex == 'M':
        sex_M = 1
    if sex == 'F':
        sex_F = 1
    if sex not in ['M', 'F']:
        sex_U = 1

    #load the sessions file for extracting sessions
    sessions_file_path = os.path.join(project_data_dir, str(participant_id), 'sessions.csv')
    sessions_file = pd.read_csv(sessions_file_path)

    #check if the participant has at least 2 sessions
    num_sessions = sessions_file.shape[0]
    if num_sessions < 2:
        print(f"Warning: Participant {participant_id} has less than 2 sessions. Skipping.")
        return None
    
    #generate block for one participant
    else:
        scan_list = []
        age_list = []
        field_strength_list = []

        #extract pairs of sessions
        for i in range(num_sessions-1):
            scan_id = sessions_file.iloc[i]['session_id']
            field_strength = check_fieldstrength(participant_id, scan_id, folder_path)
            scan_session = sessions_file[sessions_file['session_id'] == scan_id]
            age = scan_session.iloc[0]['age']

            if np.isnan(age): #skip if age is not available
                print(f"No age available for participant {participant_id} in session {scan_id}. Skipping this session.")
                continue


            if field_strength is None:
                print(f"Warning: Field strength not found for participant {participant_id} in session {scan_id}. Skipping this pair.")
                continue
            if field_strength < 2:
                print(f"Warning: Participant {participant_id} has a field strength below 2 Tesla in session {scan_id}. Skipping this pair.")
                continue

            scan_list.append(scan_id)
            age_list.append(age)
            field_strength_list.append(field_strength)


        return pd.DataFrame({
            'participant_id': [participant_id] * len(scan_list),
            'sex_M': [sex_M] * len(scan_list),
            'sex_F': [sex_F] * len(scan_list),
            'sex_U': [sex_U] * len(scan_list),
            'age': age_list,
            'session_id': scan_list,
            'field_strength': field_strength_list
        })



class loader3D(Dataset):
    """
    Args:
        participant_df: dataframe with basic participant data (ids and gender)
        data_directory: path to the data directory
        image_size: size of the input image
        target_name: name of the target variable
        optional_meta: list of optional metadata features
    """
    
    def __init__(self, args, participant_df):

        #store all blocks of pairs from one participant
        blocks = []

        df = participant_df 

        for _, row in df.iterrows():
            participant_id = str(row['participant_id'])
            sex = str(row['sex'])
            block = build_participant_block(participant_id, sex, folder_path=args.data_directory, project_data_dir=args.project_data_dir)
            if block is not None:
                blocks.append(block)

        # concatenate all blocks into one dataframe
        self.demo = pd.concat(blocks, ignore_index=True)
        
        self.image_size = args.image_size #resize images
        self.resize = tio.transforms.Resize(tuple(self.image_size)) #safe resize transform
        self.targetname = args.target_name #save target for training
        self.datadir = args.data_directory  #save data directory

        # Build file path pairs
        self.image_paths = []
        valid_demo_rows = []
        for _, row in self.demo.iterrows():
            participant_id = str(row['participant_id'])
            session = str(row['session_id'])
            img_dir = os.path.join(self.datadir, 'derivatives', 'mriprep', participant_id, session)
            pattern = os.path.join(img_dir, '*T1w.nii.gz')

            matching_files = glob.glob(pattern)

            if not matching_files:
                print(f"Warning: No matching T1w image found for {participant_id} in session(s). Skipping.")
                continue #skip if no matching files are found
            path = matching_files[0]
            self.image_paths.append(path)
            valid_demo_rows.append(row)

        self.demo = pd.DataFrame(valid_demo_rows).reset_index(drop=True)
        self.targets = self.demo[self.targetname].values

        #check length of loaded data
        print(f"Loaded {len(self.image_paths)} images.")
        print(f"Loaded {len(self.demo)} rows of metadata.")
        print(f"Loaded {len(self.targets)} targets.")

        if len(args.optional_meta)>0:
            self.optional_meta = np.array(self.demo[args.optional_meta]).astype('float32')

        else:
            self.optional_meta = np.array([])


    def __getitem__(self, index):
        # Get target as float tensor
        target = torch.tensor([self.demo[self.targetname].iloc[index]], dtype=torch.float32)

        path = self.image_paths[index]
        

        # Load images as torchio images
        image = tio.ScalarImage(path)
        image = self.resize(image)
        image_tensor = image.data

        if len(self.optional_meta) > 0:
            meta = torch.tensor(self.optional_meta[index], dtype=torch.float32)
            return [image_tensor, meta, target]

        else:
            return [image_tensor, target]

        
    def __len__(self):
        return len(self.image_paths)