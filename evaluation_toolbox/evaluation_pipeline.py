import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--results_file', default='blank', type=str, help="directory of the results to be evaluated")

    args = parser.parse_args()

    return args


def decide_type_of_result(filepath):
    print("Filename (results): ", filepath)
    if 'results_train' in filepath.lower():
        name = 'train'
    elif 'results_val' in filepath.lower():
        name = 'val'
    elif 'results_test' in filepath.lower():
        name = 'test'
    elif 'results_ci' in filepath.lower():
        name = 'CI'
    elif 'results_all_folds' in filepath.lower():
        name = 'all_folds'
    else:
        print("Error in naming of file.")
        name = 'error'
    print(f"We are looking at {name} results.")
    return name


def scatter_plot_pred_target(predictions, targets, save_path, sex, name):
    plt.figure(figsize=(6, 6))
    plt.scatter(predictions, targets, alpha=0.5)
    min_val = min(predictions.min(), targets.min())
    max_val = max(predictions.max(), targets.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x (optimal for healthy participants)')
    plt.xlabel('Prediction (years)')
    plt.ylabel('Target (years)')
    plt.title('Target vs Prediction (' + name + '), ' + sex + ' participants')
    plt.grid(True)
    plt.legend
    plt.tight_layout()
    plot_path = os.path.join(save_path, sex + '_scatter_plot_pred_target.png')
    plt.savefig(plot_path)
    plt.close()


def scatter_plot_ages_residuals(ages, residuals, save_path, sex, name):
    plt.figure(figsize=(6, 6))
    plt.scatter(ages, residuals, alpha=0.5)
    min_val = min(ages)
    max_val = max(ages)
    plt.plot([min_val, max_val], [0, 0], 'r--', label='y = 0 (optimal)')
    plt.xlabel('Age (years)')
    plt.ylabel('Residual')
    plt.title('Age vs Residual (' + name + '), ' + sex + ' participants')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(save_path, sex + '_scatter_plot_age_residuals.png')
    plt.savefig(plot_path)
    plt.close()


def scatter_plot_targets_residuals(targets, residuals, save_path, sex, name):
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, residuals, alpha=0.5)
    min_val = min(targets)
    max_val = max(targets)
    plt.plot([min_val, max_val], [0, 0], 'r--', label='y = 0 (optimal)')
    plt.xlabel('Target (years)')
    plt.ylabel('Residual')
    plt.title('Targets vs Residual (' + name + '), ' + sex + ' participants')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(save_path, sex + '_scatter_plot_targets_residuals.png')
    plt.savefig(plot_path)
    plt.close()


def scatter_plot_targets_pace(targets, pace, save_path, name):
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, pace, alpha=0.5)
    min_val = min(targets)
    max_val = max(targets)
    plt.plot([min_val, max_val], [1, 1], 'r--', label='y = 1 (optimal)')
    plt.xlabel('Target (years)')
    plt.ylabel('Pace of ageing')
    plt.title('Targets vs Pace of ageing (' + name + ')')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(save_path, 'scatter_plot_targets_pace.png')
    plt.savefig(plot_path)
    plt.close()


def boxplot_residuals_by_agegroup(ages, residuals, save_path, sex, name):
    # Fixed bins: <60, 60-65, 65-70, >=70
    bins = [0, 60, 65, 70, 120]
    labels = ['<60', '60-65', '65-70', '70+']
    age_bins = pd.cut(ages, bins=bins, labels=labels, right=False)
    
    df = pd.DataFrame({'AgeBin': age_bins, 'Residuals': residuals})
    
    # Plot
    plt.figure(figsize=(8, 6))
    df.boxplot(column='Residuals', by='AgeBin', grid=False)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Age Group')
    plt.ylabel('Residual (Target - Prediction)')
    plt.title(f'Residuals by Age Group ({name}, {sex} participants)')
    plt.suptitle('')
    plt.tight_layout()
    
    # Save
    plot_path = os.path.join(save_path, f'{sex}_boxplot_residuals_by_agegroup.png')
    plt.savefig(plot_path)
    plt.close()

def boxplot_residuals_by_agegroup2(ages, residuals, save_path, sex, name):
    # Fixed bins: <60, 60-65, 65-70, >=70
    bins = [0, 60, 65, 70, 120]
    labels = ['<60', '60-65', '65-70', '70+']
    age_bins = pd.cut(ages, bins=bins, labels=labels, right=False)

    df = pd.DataFrame({'AgeBin': age_bins, 'Residuals': residuals})

    # Create the boxplot and capture the artists to find outliers
    plt.figure(figsize=(10, 6))
    ax = df.boxplot(column='Residuals', by='AgeBin', grid=False, return_type='axes')['Residuals']

    # Add horizontal line for zero residual
    plt.axhline(0, color='red', linestyle='--')

    # Count total points and outliers per bin
    counts = df.groupby('AgeBin').size()

    # Calculate outliers manually
    outlier_counts = {}
    for i, label in enumerate(labels):
        data = df[df['AgeBin'] == label]['Residuals'].dropna()
        if len(data) == 0:
            outlier_counts[label] = 0
            continue
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_counts[label] = len(outliers)

    # Annotate with counts
    for i, label in enumerate(labels):
        n_total = counts.get(label, 0)
        n_outliers = outlier_counts.get(label, 0)
        text = f'n={n_total}\noutl={n_outliers}'
        plt.text(i + 1, plt.ylim()[1] * 0.9, text, ha='center', fontsize=9, color='black')

    # Final plot settings
    plt.xlabel('Age Group')
    plt.ylabel('Residual (Target - Prediction)')
    plt.title(f'Residuals by Age Group ({name}, {sex} participants)')
    plt.suptitle('')
    plt.tight_layout()

    # Save
    plot_path = os.path.join(save_path, f'2{sex}_boxplot_residuals_by_agegroup.png')
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":

    """
    Setup and prepare data and directories.
    """
    args = parse_args()

    name = decide_type_of_result(args.results_file)

    results = pd.read_csv(args.results_file)
    
    #prepare data directory for saving outputs.
    dir = os.path.dirname(args.results_file)
    folder_name = "evaluation_" + name
    save_path = os.path.join(dir,folder_name)
    os.makedirs(save_path, exist_ok=True)

    idx_M = results[results['Sex (M)'] == 1].index
    idx_F = results[results['Sex (F)'] == 1].index
    idx_mt = results[results['Target']>=1].index
    idx_lt = results[results['Target']>=5].index
    #potentially add in sex undefined

    predictions = results['Prediction'].values
    targets = results['Target'].values
    residuals = targets - predictions
    ages = results['Age'].values
    sex = results['Sex (M)'].values
    pace = predictions / targets

    #save results in eval_dict to export later
    eval_dict = {
        'group': ['All', 'M', 'F']
    }


    """
    Analysis between predictions and duration (target).
    """
    r,p = pearsonr(predictions, targets)
    r_M,p_M = pearsonr(predictions[idx_M], targets[idx_M])
    r_F,p_F = pearsonr(predictions[idx_F], targets[idx_F])
    eval_dict['PCC(pred,target)'] = [r, r_M, r_F]
    eval_dict['p(pred,target)'] = [p, p_M, p_F]
    scatter_plot_pred_target(predictions, targets, save_path, 'all', name)
    scatter_plot_pred_target(predictions[idx_M], targets[idx_M], save_path, 'M', name)
    scatter_plot_pred_target(predictions[idx_F], targets[idx_F], save_path, 'F', name)
    mae = np.mean(np.abs(residuals))
    mae_M = np.mean(np.abs(residuals[idx_M]))
    mae_F = np.mean(np.abs(residuals[idx_F]))
    sd_mae = np.std(np.abs(residuals))
    sd_mae_M = np.std(np.abs(residuals[idx_M]))
    sd_mae_F = np.std(np.abs(residuals[idx_F]))
    mean_res = np.mean(residuals)
    mean_res_M = np.mean(residuals[idx_M])
    mean_res_F = np.mean(residuals[idx_F])
    eval_dict['MAE'] = [mae, mae_M, mae_F]
    eval_dict['SD MAE'] = [sd_mae, sd_mae_M, sd_mae_F]
    eval_dict['mean res'] = [mean_res, mean_res_M, mean_res_F]


    """
    Analysis between age and residuals.
    """
    r,p = pearsonr(ages, residuals)
    r_M, p_M = pearsonr(ages[idx_M], residuals[idx_M])
    r_F, p_F = pearsonr(ages[idx_F], residuals[idx_F])
    eval_dict['PCC(age,res)'] = [r, r_M, r_F]
    eval_dict['p(age,res)'] = [p, p_M, p_F]
    scatter_plot_ages_residuals(ages, residuals, save_path, 'all', name)
    scatter_plot_ages_residuals(ages[idx_M], residuals[idx_M], save_path, 'M', name)
    scatter_plot_ages_residuals(ages[idx_F], residuals[idx_F], save_path, 'F', name)


    """
    Analysis between duration (target) and residuals.
    """
    r,p = pearsonr(targets, residuals)
    r_M, p_M = pearsonr(targets[idx_M], residuals[idx_M])
    r_F, p_F = pearsonr(targets[idx_F], residuals[idx_F])
    eval_dict['PCC(target,res)'] = [r, r_M, r_F]
    eval_dict['p(target,res)'] = [p, p_M, p_F]
    scatter_plot_targets_residuals(targets, residuals, save_path, 'all', name)
    scatter_plot_targets_residuals(targets[idx_M], residuals[idx_M], save_path, 'M', name)
    scatter_plot_targets_residuals(targets[idx_F], residuals[idx_F], save_path, 'F', name)


    """
    Analysis between sex and residuals.
    """
    r,p = pearsonr(sex, residuals)
    eval_dict['PCC(sex(M),res)'] = [r, np.nan, np.nan]
    eval_dict['p(sex(M),res)'] = [p, np.nan, np.nan]


    """
    Analysis of pace of ageing.
    """
    eval_dict['Mean pace (all,>=1,>=5)'] = [np.mean(pace), np.mean(pace[idx_mt]), np.mean(pace[idx_lt])]
    eval_dict['SD pace (all,>=1,>=5)'] = [np.std(pace), np.std(pace[idx_mt]), np.std(pace[idx_lt])]
    r,p = pearsonr(pace[idx_mt], targets[idx_mt])
    eval_dict['PCC(pace,target), >= 1y'] = [r, np.nan, np.nan]
    eval_dict['p(pace,target), >= 1y'] = [p, np.nan, np.nan]
    scatter_plot_targets_pace(targets[idx_mt], pace[idx_mt], save_path, name + ', >=1 years')


    """
    Playground for future analysis.
    """
    
    adj_residuals= residuals[idx_lt]/targets[idx_lt]
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
    evaluation_results.to_csv(os.path.join(save_path, f'{name}_evaluation.csv'))

    boxplot_residuals_by_agegroup(ages, residuals, save_path, 'all', name)
    boxplot_residuals_by_agegroup2(ages, residuals, save_path, 'all', name)