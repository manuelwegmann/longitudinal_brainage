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
    error_F = np.abs(df_F['Prediction'] - df_F['Target'])

    # Test with absolute errors for whole dataset
    stat, p = stats.mannwhitneyu(error_M, error_F)
    print(f"{p:.4f} -> {'different' if p < 0.05 else 'same'} distribution for {model_name} for whole dataset with absolute errors.")


analyse_model(pred_CS, "CS CNN")
analyse_model(pred_LILAC, "LILAC")
analyse_model(pred_LILACp, "LILAC+")
analyse_model(pred_AE, "AEM-4 (age)")