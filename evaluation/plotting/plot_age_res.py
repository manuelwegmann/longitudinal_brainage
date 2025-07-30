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
fig, axes = get_figures(n_rows=1, n_cols=3, figsize=(4, 4), sharex=True, sharey=True)

"""
Load Data
"""
results_LILAC = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_final_run/predictions_all_folds.csv')
results_LILACp = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_plus_final_run/predictions_all_folds.csv')
results_CS = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/CS_final_run/fold_0/longitudinal_predictions.csv')

age_LILAC = results_LILAC['Age']
targets_LILAC = results_LILAC['Target']
preds_LILAC = results_LILAC['Prediction']
res_LILAC = targets_LILAC - preds_LILAC

age_CS = results_CS['Age']
targets_CS = results_CS['Target']
preds_CS = results_CS['Prediction']
res_CS = targets_CS - preds_CS

age_LILACp= results_LILACp['Age']
targets_LILACp = results_LILACp['Target']
preds_LILACp = results_LILACp['Prediction']
res_LILACp = targets_LILACp - preds_LILACp


"""
LILAC
"""

"""Age vs Residual"""
axes[0].scatter(age_LILAC, res_LILAC, alpha=0.5, color='steelblue')
axes[0].set_xlabel('Age [years]')
axes[0].set_ylabel('Residual [years]')
axes[0].set_title('LILAC')

# Fit regression line
coeffs = np.polyfit(age_LILAC, res_LILAC, deg=1)
line = np.poly1d(coeffs)
x_vals = np.linspace(age_LILAC.min(), age_LILAC.max(), 100)

# Plot ideal and regression line
axes[0].plot(x_vals, np.zeros(len(x_vals)),  color='black', linestyle='--', linewidth=1.2, label='Ideal')
axes[0].plot(x_vals, line(x_vals), color='red', linewidth=1.5, label = 'Fit')
axes[0].legend(loc='upper left')




"""
CS
"""
"""Age vs Residual"""
axes[1].scatter(age_CS, res_CS, alpha=0.5, color='steelblue')
axes[1].set_xlabel('Age [years]')
axes[1].set_title('CS CNN')

# Fit regression line
coeffs = np.polyfit(age_CS, res_CS, deg=1)
line = np.poly1d(coeffs)
x_vals = np.linspace(age_CS.min(), age_CS.max(), 100)

# Plot ideal and regression line
axes[1].plot(x_vals, np.zeros(len(x_vals)),  color='black', linestyle='--', linewidth=1.2, label='Ideal')
axes[1].plot(x_vals, line(x_vals), color='red', linewidth=1.5, label = 'Fit')


"""
LILAC
"""

"""Age vs Residual"""
axes[2].scatter(age_LILACp, res_LILACp, alpha=0.5, color='steelblue')
axes[2].set_xlabel('Age [years]')
axes[2].set_title('LILAC+')

# Fit regression line
coeffs = np.polyfit(age_LILACp, res_LILACp, deg=1)
line = np.poly1d(coeffs)
x_vals = np.linspace(age_LILACp.min(), age_LILACp.max(), 100)

# Plot ideal and regression line
axes[2].plot(x_vals, np.zeros(len(x_vals)),  color='black', linestyle='--', linewidth=1.2, label='Ideal')
axes[2].plot(x_vals, line(x_vals), color='red', linewidth=1.5, label = 'Fit')


"""
Style Axes, Resize and Safe
"""

# Style both axes
fig = set_style_ax(fig, axes)

# Optionally resize if needed (here it matches figsize, but just to show usage)
fig = set_size(fig, 18, 6)

os.makedirs("figures", exist_ok=True)
save_figure(fig, "figures/age_res.png")

fig.show()