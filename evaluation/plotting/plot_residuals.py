import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
results_CS = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/CS_CNN/longitudinal_predictions.csv')
results_AE = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/AE_age_4/predictions_all_folds.csv')

age_LILAC = results_LILAC['Age']
targets_LILAC = results_LILAC['Target']
preds_LILAC = results_LILAC['Prediction']
res_LILAC = targets_LILAC - preds_LILAC
mae_LILAC = np.abs(res_LILAC)

age_CS = results_CS['Age']
targets_CS = results_CS['Target']
preds_CS = results_CS['Prediction']
res_CS = targets_CS - preds_CS
mae_CS = np.abs(res_CS)

age_LILACp= results_LILACp['Age']
targets_LILACp = results_LILACp['Target']
preds_LILACp = results_LILACp['Prediction']
res_LILACp = targets_LILACp - preds_LILACp
mae_LILACp = np.abs(res_LILACp)

# Residuals
residuals = {
    'LILAC': res_LILAC,
    'LILAC+': res_LILACp,
    'CS CNN': res_CS
}
model_names = list(residuals.keys())
colors = ['blue', 'red', 'green']

# Histogram binning
bins = np.linspace(-12, 12, 30)
bin_centres = 0.5 * (bins[1:] + bins[:-1])
bin_width = (bins[1] - bins[0]) * 0.25

# Manual histogram counts
counts = [np.histogram(residuals[name], bins=bins)[0] for name in model_names]

# Create 1x2 figure
fig, axes = get_figures(n_rows=1, n_cols=2, figsize=(7, 3.5), sharex=False, sharey=False)

# Left: grouped histogram
ax0 = axes[0]
for i, (name, color) in enumerate(zip(model_names, colors)):
    offset = (i - 1) * bin_width
    ax0.bar(bin_centres + offset, counts[i], width=bin_width, color=color, alpha=0.7, label=name)

ax0.set_title('Residual Histogram')
ax0.set_xlabel('Residual')
ax0.set_ylabel('Count')
ax0.axvline(0, color='k', linestyle='--', linewidth=1)
ax0.legend(loc='upper left')

# Right: KDE plot
ax1 = axes[1]
for name, color in zip(model_names, colors):
    sns.kdeplot(
        residuals[name],
        ax=ax1,
        color=color,
        linewidth=1.8,
        label=name,
        bw_adjust=1.2,
        clip=[-12, 12],
        fill=True,
        alpha=0.35  # transparency of the fill
    )
ax1.set_title('Residual Density')
ax1.set_xlabel('Residual')
ax1.set_ylabel('Density')
ax1.axvline(0, color='k', linestyle='--', linewidth=1)
# Style, size, save
fig = set_style_ax(fig, axes, both_axes=False)
fig = set_size(fig, 20, 10)
os.makedirs("figures", exist_ok=True)
save_figure(fig, "figures/residual_histogram_density.png")
plt.show()

