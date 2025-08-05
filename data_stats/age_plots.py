import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import argparse
import json
from argparse import Namespace

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
plotting_path =os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evaluation'))
sys.path.append(scripts_path)
sys.path.append(plotting_path)
from loader import loader3D as loader
from loader_CS import loader3D as cs_loader
from plotting import (
    set_r_params,
    get_figures,
    set_style_ax,
    set_size,
    save_figure,
)

#function to load model parameters
def load_args_from_json(filepath):

    with open(filepath, 'r') as f:
        args_dict = json.load(f)

    args = Namespace(**args_dict)

    return args

args = load_args_from_json('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC/run_details.json')
args_cs = load_args_from_json('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/CS_CNN/run_details.json')

participants = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/participants.csv')
participants_cs = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/participants_cs.csv')

data = loader(args, participants[['participant_id', 'sex']]).demo
data_cs = cs_loader(args_cs, participants_cs[['participant_id', 'sex']]).demo
data['sex'] = np.where(data['sex_M'] == 1, 'Male', 'Female')
data_cs['sex'] = np.where(data_cs['sex_M'] == 1, 'Male', 'Female')

#
set_r_params(small = 8)

# Create subplots: 1 row, 2 columns
fig, axes = get_figures(n_rows=1, n_cols=2, figsize=(6, 6), sharex=False, sharey=True)

# First plot — LILAC dataset
sns.histplot(data=data, x='age', hue='sex', multiple='stack', bins=15, ax=axes[0])
axes[0].set_title('Longitudinal Data Age Histogram')
axes[0].set_xlabel('Age (baseline)')
axes[0].set_ylabel('Count')

# Second plot — CS_CNN dataset
sns.histplot(data=data_cs, x='age', hue='sex', multiple='stack', bins=15, ax=axes[1])
axes[1].set_title('Cross-Sectional Data Age Histogram')
axes[1].set_xlabel('Age')
axes[1].legend_.remove()

fig = set_style_ax(fig, axes, both_axes=False)
fig = set_size(fig, 6, 3)
os.makedirs('figures', exist_ok=True)
save_figure(fig, f'figures/age_histograms.png')

# KDE plot: Age distribution by sex (side-by-side)
fig_kde, axes_kde = get_figures(n_rows=1, n_cols=2, figsize=(6, 6), sharex=False, sharey=True)

# KDE for LILAC
sns.kdeplot(data=data, x='age', hue='sex', ax=axes_kde[0], fill=True, common_norm=False, linewidth=1.5)
axes_kde[0].set_title('Longitudinal Data Age Density')
axes_kde[0].set_xlabel('Age (baseline)')
axes_kde[0].set_ylabel('Density')

# KDE for CS_CNN
sns.kdeplot(data=data_cs, x='age', hue='sex', ax=axes_kde[1], fill=True, common_norm=False, linewidth=1.5)
axes_kde[1].set_title('Cross-Sectional Data Age Density')
axes_kde[1].set_xlabel('Age')
axes_kde[1].set_ylabel('')
axes_kde[1].legend_.remove()

# Finalise styling
fig_kde = set_style_ax(fig_kde, axes_kde, both_axes=False)
fig_kde = set_size(fig_kde, 6, 3)
save_figure(fig_kde, 'figures/age_kde_plots')
plt.show()




