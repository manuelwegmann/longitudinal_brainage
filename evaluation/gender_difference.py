import numpy as np
import pandas as pd
import scipy.stats as stats
import os

"""
Load Data
"""
pred_CS = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/CS_CNN/longitudinal_predictions.csv')
pred_LILAC = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC/predictions_all_folds.csv')
pred_LILACp = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_plus/predictions_all_folds.csv')
pred_AE = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/AE_age_4/predictions_all_folds.csv')


def analyse_model(df, model_name):
    df_M = df[df['Sex (M)'] == 1]
    df_F = df[df['Sex (M)'] == 0]

    error_M = np.abs(df_M['Prediction'] - df_M['Target'])
    error_adj_M = error_M / df_M['Target']
    error_F = np.abs(df_F['Prediction'] - df_F['Target'])
    error_adj_F = error_F / df_F['Target']

    # short term
    idx_st_M = df_M[df_M['Target'] < 1].index
    idx_st_F = df_F[df_F['Target'] < 1].index
    stat, p = stats.mannwhitneyu(error_M.loc[idx_st_M], error_F.loc[idx_st_F])
    print(f"{p:.4f} -> {'different' if p < 0.05 else 'same'} distribution for {model_name} short term.")

    # long term
    idx_lt_M = df_M[df_M['Target'] >= 1].index
    idx_lt_F = df_F[df_F['Target'] >= 1].index
    stat, p = stats.mannwhitneyu(error_adj_M.loc[idx_lt_M], error_adj_F.loc[idx_lt_F])
    print(f"{p:.4f} -> {'different' if p < 0.05 else 'same'} distribution for {model_name} long term.")

analyse_model(pred_CS, "CS CNN")
analyse_model(pred_LILAC, "LILAC")
analyse_model(pred_LILACp, "LILAC+")
analyse_model(pred_AE, "AEM-4 (age)")