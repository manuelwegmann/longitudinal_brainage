import sys
import os
import argparse
from argparse import Namespace
import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from plotting import (
    set_r_params,
    get_figures,
    set_style_ax,
    set_size,
    save_figure,
)

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
sys.path.append(scripts_path)
from loader import loader3D

#function to load model parameters
def load_args_from_json(filepath):

    with open(filepath, 'r') as f:
        args_dict = json.load(f)

    args = Namespace(**args_dict)

    return args

participants = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/participants.csv')
args = load_args_from_json('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC/run_details.json')

data = loader3D(args, participants).demo
print(data.head())
age = data['age'].values
target = data['duration'].values
r,p = pearsonr(age,target)
print(f"Pearson's r for age and target: {r}, p-value: {p}")

# Set plotting parameters (font, sizes, etc.)
set_r_params()
# Create figure with shared axes
fig, axes = get_figures(n_rows=1, n_cols=1, figsize=(4, 4), sharex=True, sharey=True)

axes.scatter(age, target, alpha=0.5, color='steelblue')
axes.set_xlabel('Age [years]')
axes.set_ylabel('Target [years]')
axes.set_title('Age at Baseline versus Duration between Scans')

# Fit regression line
coeffs = np.polyfit(age, target, deg=1)
line = np.poly1d(coeffs)
x_vals = np.linspace(age.min(), age.max(), 100)

# Plot ideal and regression line
axes.plot(x_vals, line(x_vals), color='red', linewidth=1.5, label = 'Fit')
axes.legend(loc='upper left')

# Style both axes
fig = set_style_ax(fig, np.array([axes]))

# Optionally resize if needed (here it matches figsize, but just to show usage)
fig = set_size(fig, 12, 12)

os.makedirs("figures", exist_ok=True)
save_figure(fig, "figures/cor_age_target.png")

fig.show()