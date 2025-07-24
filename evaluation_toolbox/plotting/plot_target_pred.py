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
fig, axes = get_figures(n_rows=1, n_cols=3, figsize=(12, 4), sharex=True, sharey=True)

"""
Load Data
"""
results_LILAC = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_final_run/predictions_all_folds.csv')
results_LILACp = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_plus_final_run/predictions_all_folds.csv')
results_CS = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/CS_final_run/fold_0/longitudinal_predictions.csv')

targets_LILAC = results_LILAC['Target']
preds_LILAC = results_LILAC['Prediction']

targets_LILACp = results_LILACp['Target']
preds_LILACp = results_LILACp['Prediction']

targets_CS = results_CS['Target']
preds_CS = results_CS['Prediction']


"""
LILAC
"""
axes[0].scatter(targets_LILAC, preds_LILAC, alpha=0.5, color='steelblue')
axes[0].set_title('LILAC')
axes[0].set_xlabel('Target [years]')
axes[0].set_ylabel('Prediction [years]')

# Fit regression line
coeffs_LILAC = np.polyfit(targets_LILAC, preds_LILAC, deg=1)
line_LILAC = np.poly1d(coeffs_LILAC)
x_vals = np.linspace(targets_LILAC.min(), targets_LILAC.max(), 100)

# Plot ideal and regression line
axes[0].plot(x_vals, x_vals,  color='black', linestyle='--', linewidth=1.2, label='Ideal')
axes[0].plot(x_vals, line_LILAC(x_vals), color='red', linewidth=1.5, label = 'Fit')
axes[0].legend(loc='upper left')


"""
CS
"""
axes[1].scatter(targets_CS, preds_CS, alpha=0.5, color='steelblue')
axes[1].set_title('CS CNN')
axes[1].set_xlabel('Target [years]')

# Fit regression line
coeffs_CS = np.polyfit(targets_CS, preds_CS, deg=1)
line_CS = np.poly1d(coeffs_CS)
x_vals = np.linspace(targets_CS.min(), targets_CS.max(), 100)

# Plot ideal and regression line
axes[1].plot(x_vals, x_vals,  color='black', linestyle='--', linewidth=1.2, label='Ideal')
axes[1].plot(x_vals, line_CS(x_vals), color='red', linewidth=1.5, label = 'Fit')


"""
LILAC+
"""
axes[2].scatter(targets_LILACp, preds_LILACp, alpha=0.5, color='steelblue')
axes[2].set_title('LILAC+')
axes[2].set_xlabel('Target [years]')

# Fit regression line
coeffs_LILACp = np.polyfit(targets_LILACp, preds_LILACp, deg=1)
line_LILACp = np.poly1d(coeffs_LILACp)
x_vals = np.linspace(targets_LILACp.min(), targets_LILACp.max(), 100)

# Plot ideal and regression line
axes[2].plot(x_vals, x_vals,  color='black', linestyle='--', linewidth=1.2, label='Ideal')
axes[2].plot(x_vals, line_LILACp(x_vals), color='red', linewidth=1.5, label = 'Fit')


"""
Style Axes, Resize and Safe
"""

# Style both axes
fig = set_style_ax(fig, axes)

# Optionally resize if needed (here it matches figsize, but just to show usage)
fig = set_size(fig, 18, 6)

os.makedirs("figures", exist_ok=True)
save_figure(fig, "figures/target_pred.png")

fig.show()