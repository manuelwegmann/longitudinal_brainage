import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from plotting import (
    set_r_params,
    get_figures,
    set_style_ax,
    set_size,
    save_figure,
)

# Set plotting parameters (font, sizes, etc.)
set_r_params()

# Load data
results_LILAC = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_final_run/predictions_all_folds.csv')
results_LILACp = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_plus_final_run/predictions_all_folds.csv')
results_CS = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/CS_final_run/fold_0/longitudinal_predictions.csv')

# Compute residuals
results_LILAC['Residual'] = results_LILAC['Target'] - results_LILAC['Prediction']
results_CS['Residual'] = results_CS['Target'] - results_CS['Prediction']
results_LILACp['Residual'] = results_LILACp['Target'] - results_LILACp['Prediction']

# Define age bins and labels
bins = [0, 60, 65, 70, 200]
labels = ['<60', '60-65', '65-70', '70+']

# Bin ages
results_LILAC['AgeBin'] = pd.cut(results_LILAC['Age'], bins=bins, labels=labels, right=False)
results_CS['AgeBin'] = pd.cut(results_CS['Age'], bins=bins, labels=labels, right=False)
results_LILACp['AgeBin'] = pd.cut(results_LILACp['Age'], bins=bins, labels=labels, right=False)

# Create figure and axes: 1 row, 3 cols
fig, axes = get_figures(n_rows=1, n_cols=3, figsize=(15,6), sharey=True)

def boxplot_by_age(ax, df, title):
    data_to_plot = [df.loc[df['AgeBin'] == age_bin, 'Residual'].dropna() for age_bin in labels]
    ax.boxplot(data_to_plot, patch_artist=True, boxprops=dict(facecolor='steelblue', alpha=0.7))
    ax.set_xticks([1, 2, 3, 4])         # Fix: set tick positions to match boxplot positions
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel('Age range [years]')
    ax.grid(axis='y', color='C7', linestyle='--', lw=.8)
    
boxplot_by_age(axes[0], results_LILAC, 'LILAC')
axes[0].set_ylabel('Residual [years]')

boxplot_by_age(axes[1], results_CS, 'CS CNN')

boxplot_by_age(axes[2], results_LILACp, 'LILAC+')

# Style axes, resize figure, and save
fig = set_style_ax(fig, axes, both_axes=False)
fig = set_size(fig, 15, 6)

os.makedirs("figures", exist_ok=True)
save_figure(fig, "figures/residual_boxplots_by_age.png")

plt.show()