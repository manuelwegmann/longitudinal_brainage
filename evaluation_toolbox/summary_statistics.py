import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse
import scipy.stats as stats


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--predictions_file', default='blank', type=str, help="directory of the results to be evaluated")

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    """
    Load data.
    """
    args = parse_args()

    results = pd.read_csv(args.predictions_file)

    idx_M = results[results['Sex (M)'] == 1].index
    idx_F = results[results['Sex (M)'] == 0].index
    idx_st = results[(results['Target'] >= 1) & (results['Target'] < 3)].index
    idx_mt = results[(results['Target'] >= 3) & (results['Target'] < 5)].index
    idx_lt = results[results['Target']>=5].index

    preds = results['Prediction'].values
    targets = results['Target'].values
    res = targets - preds
    ages = results['Age'].values
    paces = preds / targets

    #save results in eval_dict to export later
    eval_dict = {
        'group': ['All', 'M', 'F']
    }


    """
    Analysis between predictions and targets.
    """
    mae = np.mean(np.abs(res))
    mae_M = np.mean(np.abs(res[idx_M]))
    mae_F = np.mean(np.abs(res[idx_F]))
    sd_mae = np.std(np.abs(res))
    sd_mae_M = np.std(np.abs(res[idx_M]))
    sd_mae_F = np.std(np.abs(res[idx_F]))
    mean_res = np.mean(res)
    mean_res_M = np.mean(res[idx_M])
    mean_res_F = np.mean(res[idx_F])
    eval_dict['MAE'] = [mae, mae_M, mae_F]
    eval_dict['SD MAE'] = [sd_mae, sd_mae_M, sd_mae_F]
    eval_dict['mean res'] = [mean_res, mean_res_M, mean_res_F]
    r,p = pearsonr(preds, targets)
    r_M,p_M = pearsonr(preds[idx_M], targets[idx_M])
    r_F,p_F = pearsonr(preds[idx_F], targets[idx_F])
    eval_dict['PCC(pred,target)'] = [r, r_M, r_F]
    eval_dict['p(pred,target)'] = [p, p_M, p_F]


    """
    Analysis between age and residuals.
    """
    r,p = pearsonr(ages, res)
    r_M, p_M = pearsonr(ages[idx_M], res[idx_M])
    r_F, p_F = pearsonr(ages[idx_F], res[idx_F])
    eval_dict['PCC(age,res)'] = [r, r_M, r_F]
    eval_dict['p(age,res)'] = [p, p_M, p_F]


    """
    Analysis between duration (target) and residuals.
    """
    r,p = pearsonr(targets, res)
    r_M, p_M = pearsonr(targets[idx_M], res[idx_M])
    r_F, p_F = pearsonr(targets[idx_F], res[idx_F])
    eval_dict['PCC(target,res)'] = [r, r_M, r_F]
    eval_dict['p(target,res)'] = [p, p_M, p_F]


    """
    Analysis between sex and residuals.
    """
    stat, p = stats.mannwhitneyu(res[idx_M], res[idx_F], alternative='two-sided')
    eval_dict['Mann-Whitney (H0=same)'] = [stat, p, np.nan]


    """
    Analysis of pace of ageing.
    """
    eval_dict['Mean pace (>=1,>=3,>=5)'] = [np.mean(paces[idx_st]), np.mean(paces[idx_mt]), np.mean(paces[idx_lt])]
    eval_dict['SD pace (all,>=1,>=5)'] = [np.std(paces[idx_st]), np.std(paces[idx_mt]), np.std(paces[idx_lt])]
    r,p = pearsonr(paces[idx_mt], targets[idx_mt])
    eval_dict['PCC(pace,target), >= 1y'] = [r, np.nan, np.nan]
    eval_dict['p(pace,target), >= 1y'] = [p, np.nan, np.nan]


    """
    Playground for future analysis.
    """
    
    adj_residuals= res[idx_lt]/targets[idx_lt]
    print(len(adj_residuals))
    r, p = pearsonr(targets[idx_lt], adj_residuals)
    print(f"Pearson correlation between targets and adjusted residuals: r={r}, p={p}")



    """
    Print results and save as .csv .
    """
    df = pd.DataFrame(eval_dict)
    evaluation_results = df.T
    evaluation_results.columns = evaluation_results.iloc[0]
    evaluation_results = evaluation_results.drop(evaluation_results.index[0])
    print(evaluation_results)
    dir = os.path.dirname(args.predictions_file)
    evaluation_results.to_csv(os.path.join(dir, 'summary_statistics.csv'))