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

def load_residuals(folder):
    results_CI = pd.read_csv(os.path.join(folder, 'predictions_CI.csv'))
    results_CN = pd.read_csv(os.path.join(folder, 'predictions_CN.csv'))
    res_CI = results_CI['Target'].values - results_CI['Prediction'].values
    res_CN = results_CN['Target'].values - results_CN['Prediction'].values
    return res_CI, res_CN

import numpy as np

def permutation_test(x, y, num_permutations=100000, statistic='mean', alternative='two-sided', seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    x = np.array(x)
    y = np.array(y)
    
    observed_diff = np.mean(x) - np.mean(y) if statistic == 'mean' else np.median(x) - np.median(y)
    
    combined = np.concatenate([x, y])
    count = 0
    perm_diffs = []

    for _ in range(num_permutations):
        np.random.shuffle(combined)
        new_x = combined[:len(x)]
        new_y = combined[len(x):]
        diff = np.mean(new_x) - np.mean(new_y) if statistic == 'mean' else np.median(new_x) - np.median(new_y)
        perm_diffs.append(diff)

        if alternative == 'two-sided':
            count += abs(diff) >= abs(observed_diff)
        elif alternative == 'greater':
            count += diff >= observed_diff
        elif alternative == 'less':
            count += diff <= observed_diff

    p_value = count / num_permutations
    return observed_diff, p_value, np.array(perm_diffs)

if __name__ == "__main__":

    opt = parse_args()
    set_r_params(small =8)

    # Load residuals for both models
    res_CI_1, res_CN_1 = load_residuals(opt.model_folder1)
    res_CI_2, res_CN_2 = load_residuals(opt.model_folder2)
    print(f"{opt.model_name1} mean residual CN: {np.mean(res_CN_1)}, CI: {np.mean(res_CI_1)}")
    print(f"{opt.model_name2} mean residual CN: {np.mean(res_CN_2)}, CI: {np.mean(res_CI_2)}")

    # Prepare plot
    fig, axes = get_figures(n_rows=1, n_cols=2, figsize=(10, 4), sharex=True, sharey=True)

    # Model 1 plot
    data1 = pd.DataFrame({
        'Residual': list(res_CI_1) + list(res_CN_1),
        'Group': ['CI'] * len(res_CI_1) + ['CN'] * len(res_CN_1)
    })
    sns.boxplot(data=data1, x='Group', y='Residual', ax=axes[0], palette='pastel')
    axes[0].set_title(opt.model_name1)
    axes[0].set_ylabel('Residual (Target - Prediction)')
    axes[0].set_xlabel('')

    # Model 2 plot
    data2 = pd.DataFrame({
        'Residual': list(res_CI_2) + list(res_CN_2),
        'Group': ['CI'] * len(res_CI_2) + ['CN'] * len(res_CN_2)
    })
    sns.boxplot(data=data2, x='Group', y='Residual', ax=axes[1], palette='pastel')
    axes[1].set_title(opt.model_name2)
    axes[1].set_ylabel('')
    axes[1].set_xlabel('')

    # Styling
    fig = set_style_ax(fig, axes, both_axes=False)
    fig = set_size(fig, 6, 3)

    os.makedirs('figures', exist_ok=True)
    save_figure(fig, f'figures/CI_boxplot_{opt.model_name1}_{opt.model_name2}.png')

    # Mann–Whitney U tests
    for model_name, res_CI, res_CN in zip(
        [opt.model_name1, opt.model_name2],
        [res_CI_1, res_CI_2],
        [res_CN_1, res_CN_2]
    ):  
        u_stat, p_value = stats.mannwhitneyu(res_CN, res_CI, alternative='greater')
        print(f"[{model_name}] Mann–Whitney U test: U = {u_stat:.2f}, p = {p_value:.4f}")

    #Logistic option
    res = np.concatenate([res_CI_1, res_CN_1])
    is_ci = np.concatenate([np.ones(len(res_CI_1)), np.zeros(len(res_CN_1))])


    import statsmodels.formula.api as smf

    # Assume you have:
    # res    → predictions or some model-derived feature (length N)
    # is_cn  → binary labels (1 = CI, 0 = CN)

    # Build DataFrame
    df = pd.DataFrame({
        "Prediction": res,
        "Group": is_ci
    })

    # Fit logistic regression: predict Group using Prediction
    model = smf.logit("Group ~ Prediction", data=df).fit(maxiter=100)

    # Summary of the model
    print(model.summary())