import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sys

plotting_path =os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evaluation'))
sys.path.append(plotting_path)
from plotting import (
    set_r_params,
    get_figures,
    set_style_ax,
    set_size,
    save_figure,
)

fold1 = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/val_fold_0.csv')
fold2 = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/val_fold_1.csv')
fold3 = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/val_fold_2.csv')
fold4 = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/val_fold_3.csv')
fold5 = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/val_fold_4.csv')

fold1_cs = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/cs_val_fold_0.csv')
fold2_cs = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/cs_val_fold_1.csv')
fold3_cs = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/cs_val_fold_2.csv')
fold4_cs = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/cs_val_fold_3.csv')
fold5_cs = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/cs_val_fold_4.csv')



#
set_r_params(small = 8)

# Create subplots: 1 row, 2 columns
fig, axes = get_figures(n_rows=1, n_cols=2, figsize=(6, 6), sharex=True, sharey=True)

# Plot KDEs for each original fold
for i, fold in enumerate([fold1, fold2, fold3, fold4, fold5]):
    sns.kdeplot(fold['age'], ax=axes[0], fill=False, label=f'Fold {i}', linewidth=1)

axes[0].set_title('Longitudinal Folds Age Density')
axes[0].set_xlabel('Baseline Age')
axes[0].set_ylabel('Density')
axes[0].legend(title='Fold')

# Plot KDEs for each CS fold
for i, fold in enumerate([fold1_cs, fold2_cs, fold3_cs, fold4_cs, fold5_cs]):
    sns.kdeplot(fold['age'], ax=axes[1], fill=False, label=f'Fold {i}', linewidth=1)

axes[1].set_title('CS Folds Age Density')
axes[1].set_xlabel('Baseline Age')
axes[1].set_ylabel('Density')


# Finalise styling
fig = set_style_ax(fig, axes, both_axes=True)
fig = set_size(fig, 6, 2.5)
save_figure(fig, 'figures/folds_age_kde_plots')
plt.show()




