import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

l1 = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_age/fold_0/training_metrics.csv')
l2 = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_age/fold_1/training_metrics.csv')
l3 = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_age/fold_2/training_metrics.csv')
l4 = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_age/fold_3/training_metrics.csv')
l5 = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_age/fold_4/training_metrics.csv')

train_loss = (l1['train_loss'].values + l2['train_loss'].values + l3['train_loss'].values + l4['train_loss'].values + l5['train_loss'].values)/5
val_loss = (l1['val_loss'].values + l2['val_loss'].values + l3['val_loss'].values + l4['val_loss'].values + l5['val_loss'].values)/5
train_mae = (l1['train_mae'].values + l2['train_mae'].values + l3['train_mae'].values + l4['train_mae'].values + l5['train_mae'].values)/5
val_mae = (l1['val_mae'].values + l2['val_mae'].values + l3['val_mae'].values + l4['val_mae'].values + l5['val_mae'].values)/5

# Plot and save training/validation loss curves
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Averaged Train/Val Loss')
plt.legend()
plt.grid(True)
plot_path = os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_age/evaluation_all_folds', 'loss_plot_trainval.png')
plt.savefig(plot_path)
plt.close()
print(f"Loss plot (Train/Val) saved to: {plot_path}")

# Plot and save training/validation loss curves
plt.figure(figsize=(10, 6))
plt.plot(train_mae, label='Training MAE')
plt.plot(val_mae, label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean absolute Error (MAE)')
plt.title('Averaged Train/Val MAE')
plt.legend()
plt.grid(True)
plot_path = os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_age/evaluation_all_folds', 'mae_plot_trainval.png')
plt.savefig(plot_path)
plt.close()
print(f"Loss plot (Train/Val) saved to: {plot_path}")