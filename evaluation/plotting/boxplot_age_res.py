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
results_LILAC = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC/predictions_all_folds.csv')
results_LILACp = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_plus/predictions_all_folds.csv')
results_CS = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/AE_age_4/predictions_all_folds.csv')

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

import seaborn as sns

def boxplot_by_age(ax, df, title, is_first_col=False):
    sns.boxplot(
        x='AgeBin', y='Residual', data=df, ax=ax,
        color='steelblue', showfliers=False,
        boxprops={'alpha': 0.7}
    )
    
    # Overlay jittered outliers
    for age_bin in labels:
        bin_data = df[df['AgeBin'] == age_bin]['Residual'].dropna()
        if bin_data.empty:
            continue
        q1, q3 = np.percentile(bin_data, [25, 75])
        iqr = q3 - q1
        lower_whisker = q1 - 1.5 * iqr
        upper_whisker = q3 + 1.5 * iqr
        outliers = bin_data[(bin_data < lower_whisker) | (bin_data > upper_whisker)]
        if not outliers.empty:
            sns.stripplot(
                x=[age_bin] * len(outliers), y=outliers, ax=ax,
                color='red', size=5, jitter=0.15,
                edgecolor='black', alpha=0.6
            )

    # Set ticks before setting labels to avoid warnings
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)

    # Font sizes from rcParams (assuming set_r_params sets these)
    title_fontsize = plt.rcParams['axes.titlesize']
    label_fontsize = plt.rcParams['axes.labelsize']
    tick_fontsize = plt.rcParams['xtick.labelsize']

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel('Age range [years]', fontsize=label_fontsize)
    if is_first_col:
        ax.set_ylabel('Residual [years]', fontsize=label_fontsize)

    ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)

    ax.grid(axis='y', color='C7', linestyle='--', lw=0.8)
    
boxplot_by_age(axes[0], results_LILAC, 'LILAC', is_first_col=True)
boxplot_by_age(axes[1], results_CS, 'CS CNN')
boxplot_by_age(axes[2], results_LILACp, 'LILAC+')

# Style axes, resize figure, and save
fig = set_style_ax(fig, axes, both_axes=False)
fig = set_size(fig, 18, 6.9)

os.makedirs("figures", exist_ok=True)
save_figure(fig, "figures/test.png")

plt.show()