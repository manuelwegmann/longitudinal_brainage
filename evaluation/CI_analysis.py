import numpy as np
import pandas as pd
import os
import argparse
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from plotting import (
    set_r_params,
    get_figures,
    set_style_ax,
    set_size,
    save_figure,
)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_CI', default='blank', type=str, help="Path to model predictions for CI.")
    parser.add_argument('--file_CN', default='blank', type=str, help="Path to model predictions for CN.")
    parser.add_argument('--model_name', default='blank', type=str, help="Name of the model (used for plots).")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    opt = parse_args()

    results_CI = pd.read_csv(opt.file_CI)
    results_CN = pd.read_csv(opt.file_CN)
    
    df_CI = results_CI[results_CI['Target']>=1]
    df_CN = results_CN[results_CN['Target']>=1]

    adj_res_CI = (df_CI['Target'] - df_CI['Prediction'])/df_CI['Target']
    adj_res_CN = (df_CN['Target'] - df_CN['Prediction'])/df_CN['Target']

    set_r_params()
    fig, axes = get_figures(n_rows=1, n_cols=2, figsize=(10, 10), sharex=False, sharey=False)

    # Prepare long-form DataFrame for boxplot
    adj_res_df = pd.DataFrame({
        'Adjusted Residuals': pd.concat([adj_res_CN, adj_res_CI], ignore_index=True),
        'Group': ['CN'] * len(adj_res_CN) + ['CI'] * len(adj_res_CI)
    })

    # Boxplot (left)
    sns.boxplot(
        data=adj_res_df,
        x='Group',
        y='Adjusted Residuals',
        ax=axes[0],
        palette=['#4c72b0', '#dd8452']
    )
    axes[0].set_ylabel('Adjusted Residuals', fontsize=8)
    axes[0].set_xlabel('Group', fontsize=8)

    # Density plot (right)
    sns.kdeplot(adj_res_CN, ax=axes[1], label='CN', color='#4c72b0', fill=True, alpha=0.5)
    sns.kdeplot(adj_res_CI, ax=axes[1], label='CI', color='#dd8452', fill=True, alpha=0.5)
    axes[1].set_xlabel('Adjusted Residuals', fontsize=8)
    axes[1].legend(fontsize=6)

    fig = set_style_ax(fig, axes, both_axes=False)
    fig = set_size(fig, 6, 3)
    os.makedirs('figures', exist_ok=True)
    save_figure(fig, f'figures/CI_{opt.model_name}.png')



    # Perform Mann–Whitney U test
    u_stat, p_value = stats.mannwhitneyu(adj_res_CN, adj_res_CI, alternative='greater')

    print(f"Mann–Whitney U test: U = {u_stat:.2f}, p = {p_value:.4f}")