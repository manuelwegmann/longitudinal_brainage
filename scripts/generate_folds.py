import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_directory', default='/mimer/NOBACKUP/groups/brainage/data/oasis3', type=str, help="directory of the data (OASIS3)")
    parser.add_argument('--project_data_dir', default ='/mimer/NOBACKUP/groups/brainage/thesis_brainage/data', type=str, help="directory with the updated session files")
    parser.add_argument('--participants_file_path', default = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/participan_files/participants_file_with_age.csv', type = str, help = 'path to participants csv.')

    parser.add_argument('--model', default='LILAC_plus', type=str, choices=['LILAC', 'LILAC_plus'], help="model to use: LILAC or LILAC_plus")

    #data preprocessing arguments
    parser.add_argument('--image_size', nargs=3, type=int, default=[128, 128, 128], help='Input image size as three integers (e.g. 128 128 128)')
    parser.add_argument('--image_channel', default=1, type=int, help="number of channels in the input image")
    parser.add_argument('--seed', default=15, type=int)

    #target and optional meta data arguments
    parser.add_argument('--target_name', default='duration', type=str, help="name of the target variable")
    parser.add_argument('--optional_meta', nargs='+', default=['sex_F', 'sex_M'], help="List of optional meta to be used in the model")
    
    #model architecture arguments
    parser.add_argument('--n_of_blocks', default=4, type=int, help="number of blocks in the encoder")
    parser.add_argument('--initial_channel', default=16, type=int, help="initial channel size after first conv")
    parser.add_argument('--kernel_size', default=3, type=int, help="kernel size")

    #training arguments
    parser.add_argument('--dropout', default=0, type=float, help="dropout rate")
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batchsize', default=16, type=int)
    parser.add_argument('--max_epoch', default=30, type=int, help="max epoch")
    parser.add_argument('--epoch', default=0, type=int, help="starting epoch")
    
    parser.add_argument('--folds', default=5, type=int, help = "number of folds for k-fold cv.")
    parser.add_argument('--output_directory', default='/mimer/NOBACKUP/groups/brainage/thesis_brainage/results', type=str, help="directory path for saving model and outputs")
    parser.add_argument('--run_name', default='test_run', type=str, help="name of the run")


    args = parser.parse_args()

    return args


def plot_age_curves(a1, a2, a3, a4, a5, save_path):
    plt.figure(figsize=(10, 6))

    sns.kdeplot(a1, color='red', label='Fold 1')
    sns.kdeplot(a2, color='green', label='Fold 2')
    sns.kdeplot(a3, color='blue', label='Fold 3')
    sns.kdeplot(a4, color='orange', label='Fold 4')
    sns.kdeplot(a5, color='yellow', label='Fold 5')


    plt.xlabel('Age at Baseline (years)')
    plt.ylabel('Density')
    plt.title('Age Distribution by Fold (Density Curves)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'age_curves_by_fold.png'))  # Added .png extension
    plt.show()


# Setup data
opt = parse_args()

participant_df = pd.read_csv(opt.participants_file_path)
    
age_values = participant_df['age'].values 

# Bin the targets into quantile-based bins
binned_age = pd.qcut(age_values, q=10, labels=False)

# Create stratified K-folds using the binned targets
skf = StratifiedKFold(n_splits=opt.folds, shuffle=True, random_state=opt.seed)
folds = list(skf.split(participant_df, binned_age))

for i, (train_idx, val_idx) in enumerate(folds):
    print("We are in fold:", i)
    train_fold = participant_df.iloc[train_idx].reset_index(drop=True)
    val_fold = participant_df.iloc[val_idx].reset_index(drop=True)
    train_fold = train_fold[['participant_id', 'sex', 'age']]
    val_fold = val_fold[['participant_id', 'sex', 'age']]
    train_fold.to_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/participant_files', f'train_fold_{i}.csv'), index=False)
    val_fold.to_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/participant_files', f'val_fold_{i}.csv'), index=False)

fold1 = pd.read_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/participant_files', 'val_fold_0.csv'), index=False)
fold2 = pd.read_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/participant_files', 'val_fold_1.csv'), index=False)
fold3 = pd.read_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/participant_files', 'val_fold_2.csv'), index=False)
fold4 = pd.read_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/participant_files', 'val_fold_3.csv'), index=False)
fold5 = pd.read_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/participant_files', 'val_fold_4.csv'), index=False)

age1 = fold1['age'].values
age2 = fold2['age'].values
age3 = fold3['age'].values
age4 = fold4['age'].values
age5 = fold5['age'].values

plot_age_curves(age1, age2, age3, age4, age5, '/mimer/NOBACKUP/groups/brainage/thesis_brainage/participant_files')

longitudinal_participant_ids = participant_df['participant_id'].unique()
all_participants = pd.read_csv('/mimer/NOBACKUP/groups/brainage/data/oasis3/participants.tsv', sep='\t')




