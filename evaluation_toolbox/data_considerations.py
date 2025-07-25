import sys
import os
import argparse
from argparse import Namespace
import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.append(scripts_path)
from loader import loader3D

#function to load model parameters
def load_args_from_json(filepath):

    with open(filepath, 'r') as f:
        args_dict = json.load(f)

    args = Namespace(**args_dict)

    return args

participants = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/participants.csv')
args = load_args_from_json('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_final_run/run_details.json')

data = loader3D(args, participants).demo
print(data.head())
age = data['age'].values
target = data['duration'].values
r,p = pearsonr(age,target)
print(f"Pearson's r for age and target: {r}, p-value: {p}")