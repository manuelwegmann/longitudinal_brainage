"""
Script to plot averaged training losses from multiple folds.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

evaluation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evaluation'))
sys.path.append(evaluation_path)

from plotting import (
    set_r_params,
    get_figures,
    set_style_ax,
    set_size,
    save_figure,
)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', default='blank', type=str, help="directory of the model.")
    parser.add_argument('--name', default ='blank', type=str, help="name of the model (for plot).")

    args = parser.parse_args()

    return args

def load_training_metrics(results_dir):
    metrics = {}
    for subfolder in os.listdir(results_dir):
        if subfolder.startswith('fold'):
            csv_path = os.path.join(results_dir, subfolder, 'training_metrics.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                metrics[subfolder] = df
            else:
                print(f"Warning: {csv_path} does not exist.")
    return metrics


def plot_and_save_avg_metrics(metrics_dict, save_dir, name=''):
    dfs = list(metrics_dict.values())
    n_epochs = min(df.shape[0] for df in dfs)
    all_metrics = np.stack([df.iloc[:n_epochs].to_numpy() for df in dfs], axis=0)

    means = all_metrics.mean(axis=0)
    stds = all_metrics.std(axis=0)
    epochs = np.arange(1, n_epochs + 1)

    set_r_params(small = 8)
    fig, ax = get_figures(n_rows = 1, n_cols = 2, figsize=(4, 3), sharex=True, sharey=False)

    #MSE
    ax[0].plot(epochs, means[:, 0], label='Train MSE', color='blue')
    ax[0].fill_between(epochs, means[:, 0] - stds[:, 0], means[:, 0] + stds[:, 0], color='blue', alpha=0.2)
    ax[0].plot(epochs, means[:, 2], label='Val MSE', color='orange')
    ax[0].fill_between(epochs, means[:, 2] - stds[:, 2], means[:, 2] + stds[:, 2], color='orange', alpha=0.2)
    ax[0].set_title(f'{name} Average MSE Across Folds')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend(loc='upper right')

    #MAE
    ax[1].plot(epochs, means[:, 1], label='Train MAE', color='green')
    ax[1].fill_between(epochs, means[:, 1] - stds[:, 1], means[:, 1] + stds[:, 1], color='green', alpha=0.2)
    ax[1].plot(epochs, means[:, 3], label='Val MAE', color='red')
    ax[1].fill_between(epochs, means[:, 3] - stds[:, 3], means[:, 3] + stds[:, 3], color='red', alpha=0.2)
    ax[1].set_title(f'{name} Average MAE Across Folds')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('MAE')
    ax[1].legend(loc='upper right')

    # Style axes
    fig = set_style_ax(fig, ax)

    # Resize
    fig = set_size(fig, 6, 2.5)
    path = os.path.join(save_dir, 'averaged_losses.png')
    plt.savefig(path, dpi=300)
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    model_dir = args.model_dir
    name = args.name
    metrics = load_training_metrics(model_dir)
    plot_and_save_avg_metrics(metrics, model_dir, name=name)