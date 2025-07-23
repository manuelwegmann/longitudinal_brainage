import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def plot_and_save_avg_metrics(metrics_dict, save_dir):
    dfs = list(metrics_dict.values())
    n_epochs = min(df.shape[0] for df in dfs)
    all_metrics = np.stack([df.iloc[:n_epochs].to_numpy() for df in dfs], axis=0)

    means = all_metrics.mean(axis=0)
    stds = all_metrics.std(axis=0)
    epochs = np.arange(1, n_epochs + 1)

    def apply_small_font(ax):
        ax.tick_params(axis='both', labelsize=6)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
        ax.title.set_size(8)
        for legend in ax.get_legend().get_texts():
            legend.set_fontsize(6)

    # --- Plot Loss ---
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(epochs, means[:, 0], label='Train Loss', color='blue')
    ax.fill_between(epochs, means[:, 0] - stds[:, 0], means[:, 0] + stds[:, 0], color='blue', alpha=0.2)
    ax.plot(epochs, means[:, 2], label='Val Loss', color='orange')
    ax.fill_between(epochs, means[:, 2] - stds[:, 2], means[:, 2] + stds[:, 2], color='orange', alpha=0.2)
    ax.set_title('Average Loss Across Folds')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=6)
    apply_small_font(ax)
    plt.tight_layout()
    loss_path = os.path.join(save_dir, 'average_loss.png')
    plt.savefig(loss_path, dpi=300)
    plt.close()

    # --- Plot MAE ---
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(epochs, means[:, 1], label='Train MAE', color='green')
    ax.fill_between(epochs, means[:, 1] - stds[:, 1], means[:, 1] + stds[:, 1], color='green', alpha=0.2)
    ax.plot(epochs, means[:, 3], label='Val MAE', color='red')
    ax.fill_between(epochs, means[:, 3] - stds[:, 3], means[:, 3] + stds[:, 3], color='red', alpha=0.2)
    ax.set_title('Average MAE Across Folds')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.legend(fontsize=6)
    apply_small_font(ax)
    plt.tight_layout()
    mae_path = os.path.join(save_dir, 'average_mae.png')
    plt.savefig(mae_path, dpi=300)
    plt.close()

    print(f"Plots saved:\n- {loss_path}\n- {mae_path}")


# Usage
model_dir = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_final_run'
metrics_dict = load_training_metrics(model_dir)
plot_and_save_avg_metrics(metrics_dict, model_dir)