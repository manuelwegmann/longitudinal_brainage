"""
Script to check for model capabilties to differentiate CN and CI groups.
"""

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

    parser.add_argument('--model_folder1', default='/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_CI', type=str, help="Path to folder with predictions on CN/CI for model 1")
    parser.add_argument('--model_name1', default='LILAC', type=str, help="Name of the first model")
    parser.add_argument('--model_folder2', default='/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_plus_CI', type=str, help="Path to folder with predictions on CN/CI for model 2")
    parser.add_argument('--model_name2', default='LILAC+', type=str, help="Name of the second model")

    return parser.parse_args()


def load_pace(folder):
    results_CI = pd.read_csv(os.path.join(folder, 'predictions_CI.csv'))
    results_CN = pd.read_csv(os.path.join(folder, 'predictions_CN.csv'))
    pace_CI = results_CI['Prediction'].values / results_CI['Target'].values
    pace_CN = results_CN['Prediction'].values / results_CN['Target'].values
    return pace_CI, pace_CN

if __name__ == "__main__":
    """
    Analysis by pace
    """

    opt = parse_args()
    set_r_params(small = 8)

    # Load residuals for both models
    pace_CI_1, pace_CN_1 = load_pace(opt.model_folder1)
    pace_CI_2, pace_CN_2 = load_pace(opt.model_folder2)
    print(f"{opt.model_name1} mean pace CN: {np.mean(pace_CN_1)}, CI: {np.mean(pace_CI_1)}")
    print(f"{opt.model_name2} mean pace CN: {np.mean(pace_CN_2)}, CI: {np.mean(pace_CI_2)}")

    # Prepare plot
    fig, axes = get_figures(n_rows=1, n_cols=2, figsize=(10, 4), sharex=True, sharey=True)

    # Model 1 plot
    data1 = pd.DataFrame({
        'Pace': list(pace_CN_1) + list(pace_CI_1),
        'Group': ['CN'] * len(pace_CN_1) + ['CI'] * len(pace_CI_1)
    })
    sns.boxplot(data=data1, x='Group', y='Pace', ax=axes[0], palette='pastel')
    axes[0].set_title(opt.model_name1)
    axes[0].set_ylabel('Pace (Prediction/Target)')
    axes[0].set_xlabel('')

    # Model 2 plot
    data2 = pd.DataFrame({
        'Pace': list(pace_CN_2) + list(pace_CI_2),
        'Group': ['CN'] * len(pace_CN_2) + ['CI'] * len(pace_CI_2)
    })
    sns.boxplot(data=data2, x='Group', y='Pace', ax=axes[1], palette='pastel')
    axes[1].set_title(opt.model_name2)
    axes[1].set_ylabel('')
    axes[1].set_xlabel('')

    # Styling
    fig = set_style_ax(fig, axes, both_axes=False)
    fig = set_size(fig, 6, 3)

    os.makedirs('figures', exist_ok=True)
    save_figure(fig, f'figures/pace_CI_boxplot_{opt.model_name1}_{opt.model_name2}.png')

    # Mann–Whitney U tests
    for model_name, pace_CI, pace_CN in zip(
        [opt.model_name1, opt.model_name2],
        [pace_CI_1, pace_CI_2],
        [pace_CN_1, pace_CN_2]
    ):  
        u_stat, p_value = stats.mannwhitneyu(pace_CI, pace_CN, alternative='greater')
        print(f"Pace [{model_name}] Mann–Whitney U test: U = {u_stat:.2f}, p = {p_value:.4f}")