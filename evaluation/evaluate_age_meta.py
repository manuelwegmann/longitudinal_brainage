import numpy as np
import pandas as pd
import scipy.stats as stats
import argparse
import os
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

    parser.add_argument('--file_wo_age', default='blank', type=str, help="Path to model predictions without age.")
    parser.add_argument('--file_w_age', default='blank', type=str, help="Path to model predictions with age.")
    parser.add_argument('--model_name', default='blank', type=str, help="Name of the model (used for plots).")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    opt = parse_args()

    pred_wo = pd.read_csv(opt.file_wo_age)
    pred_w = pd.read_csv(opt.file_w_age)

    n = len(pred_wo)
    if n != len(pred_w):
        raise ValueError("ERROR: The number of predictions does not line up.")


    ids_wo = []
    ids_w = []
    
    for i in range(n):
        participant = pred_wo.iloc[i]['Participant_ID']
        ses1 = pred_wo.iloc[i]['Session 1']
        ses2 = pred_wo.iloc[i]['Session 2']
        id = f"{participant}_{ses1}_{ses2}"
        ids_wo.append(id)

        participant = pred_w.iloc[i]['Participant_ID']
        ses1 = pred_w.iloc[i]['Session 1']
        ses2 = pred_w.iloc[i]['Session 2']
        id = f"{participant}_{ses1}_{ses2}"
        ids_w.append(id)

    # Compare IDs
    mismatch1 = set(ids_wo) - set(ids_w)
    mismatch2 = set(ids_w) - set(ids_wo)

    if mismatch1 or mismatch2:
        raise ValueError(f"Mismatch between ID sets.")

    pred_wo['ids'] = ids_wo
    pred_w['ids'] = ids_w

    res_wo = []
    res_w = []
    for i in range(n):
        id_df = pred_wo[pred_wo['ids']==ids_wo[i]]
        res = id_df.iloc[0]['Prediction'] - id_df.iloc[0]['Target']
        res_wo.append(res)

        id_df = pred_w[pred_w['ids']==ids_wo[i]]
        res = id_df.iloc[0]['Prediction'] - id_df.iloc[0]['Target']
        res_w.append(res)

    res_wo = np.array(res_wo)
    res_w = np.array(res_w)
    res_diff = res_wo - res_w
    error_wo = np.abs(res_wo)
    error_w = np.abs(res_w)
    error_diff = error_wo - error_w

    set_r_params()
    fig, axes = get_figures(n_rows=1, n_cols=2, figsize=(10, 10), sharex=False, sharey=False)
    
    stats.probplot(error_diff, dist="norm", plot=axes[0])
    axes[0].set_title(f'{opt.model_name} QQ-plot Error Difference')
    
    sns.kdeplot(error_diff, fill = True, ax = axes[1])
    axes[1].set_title(f"{opt.model_name} Error Difference Density")
    axes[1].set_xlabel(f"Error Difference [years]")
    axes[1].set_ylabel("Density")
    axes[1].axvline(0, color='black', label='x = 0', linestyle='--', linewidth=1)
    axes[1].legend()

    fig = set_style_ax(fig, axes, both_axes=False)
    fig = set_size(fig, 6, 3)
    os.makedirs('figures', exist_ok=True)
    save_figure(fig, f'figures/ae_analysis_{opt.model_name}.png')


    stat, p_normality = stats.shapiro(error_diff)
    print(f"Shapiro-Wilk p-value for normality: {p_normality:.4f}")

    print("Data normal — using paired t-test")
    t_stat, p_val = stats.ttest_rel(error_wo, error_w)
    print(f"p-value: {p_val:.4f}")
    print("Data not normal — using Wilcoxon signed-rank test")
    t_stat, p_val = stats.wilcoxon(error_wo, error_w)
    print(f"p-value: {p_val:.4f}")
