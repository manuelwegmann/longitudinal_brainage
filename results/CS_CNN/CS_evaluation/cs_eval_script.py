import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--predictions_file', default='/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/CS_CNN/CS_evaluation/cs_predictions.csv', type=str, help="directory of the results to be evaluated")

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    """
    Load data.
    """
    args = parse_args()

    results = pd.read_csv(args.predictions_file)

    preds = results['Prediction'].values
    targets = results['Target'].values
    res = targets - preds

    #save results in eval_dict to export later
    eval_dict = {
    }


    """
    Analysis between predictions and targets.
    """
    mse = mean_squared_error(preds, targets)
    mae = mean_absolute_error(preds, targets)
    eval_dict['MSE'] = [mse]
    eval_dict['MAE'] = [mae]
    r,p = pearsonr(preds, targets)
    eval_dict['r(pred,target)'] = [r]
    eval_dict['p(pred,target)'] = [p]
    R2 = r2_score(targets, preds)
    eval_dict['R2'] = [R2]


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

