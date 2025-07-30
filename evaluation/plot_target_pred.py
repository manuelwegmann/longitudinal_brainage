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
# Create figure with shared axes
fig, axes = get_figures(n_rows=3, n_cols=2, figsize=(10, 10), sharex=True, sharey=True)

"""
Load Data
"""
paths = {
    'CS CNN': '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/CS_CNN/longitudinal_predictions.csv',
    'LILAC': '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC/predictions_all_folds.csv',
    'LILAC+': '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_plus/predictions_all_folds.csv',
    'LILAC+ (age)': '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_age_plus/predictions_all_folds.csv',
    'AE4': '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/AE_4/predictions_all_folds.csv',
    'AE4 (age)': '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/AE_age_4/predictions_all_folds.csv',
}

# Load data
results = {name: pd.read_csv(path) for name, path in paths.items()}

# Extract targets and predictions
targets = {name: df['Target'] for name, df in results.items()}
preds = {name: df['Prediction'] for name, df in results.items()}

# Map model names to subplot positions (row, col)
model_order = [
    ('LILAC', (0, 1)),
    ('LILAC+', (1, 0)),
    ('LILAC+ (age)', (1, 1)),
    ('CS CNN', (0, 0)),
    ('AE4', (2, 0)),
    ('AE4 (age)', (2, 1)),
]

for model_name, (i, j) in model_order:
    ax = axes[i, j]
    t = targets[model_name]
    p = preds[model_name]

    ax.scatter(t, p, alpha=0.5, color='steelblue')
    ax.set_title(model_name)
    ax.set_xlabel('Target [years]')

    # Fit and plot regression line
    coeffs = np.polyfit(t, p, deg=1)
    fit_line = np.poly1d(coeffs)
    x_vals = np.linspace(t.min(), t.max(), 100)

    ax.plot(x_vals, x_vals, color='black', linestyle='--', linewidth=1.2, label='Ideal')
    ax.plot(x_vals, fit_line(x_vals), color='red', linewidth=1.5, label='Fit')
    


# Set appropiate axes label and legend
axes[0,0].set_ylabel('Prediction [years]')
axes[0,0].legend(loc='upper left')
axes[1,0].set_ylabel('Prediction [years]')
axes[2,0].set_ylabel('Prediction [years]')

# Style both axes
fig = set_style_ax(fig, axes)

# Optionally resize if needed (here it matches figsize, but just to show usage)
fig = set_size(fig, 6.9, 8)

os.makedirs("figures", exist_ok=True)
save_figure(fig, "figures/target_pred.png")

fig.show()