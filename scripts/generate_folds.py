import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold



def plot_age_curves(a1, a2, a3, a4, a5, save_path, save_name):
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
    plt.savefig(os.path.join(save_path, save_name))  # Added .png extension
    plt.show()


# Setup data

participant_df = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/participants.csv')
    
age_values = participant_df['age'].values 

# Bin the targets into quantile-based bins
binned_age = pd.qcut(age_values, q=10, labels=False)

# Create stratified K-folds using the binned targets
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
folds = list(skf.split(participant_df, binned_age))

for i, (train_idx, val_idx) in enumerate(folds):
    train_fold = participant_df.iloc[train_idx].reset_index(drop=True)
    val_fold = participant_df.iloc[val_idx].reset_index(drop=True)
    train_fold = train_fold[['participant_id', 'sex', 'age']]
    val_fold = val_fold[['participant_id', 'sex', 'age']]
    train_fold.to_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds', f'train_fold_{i}.csv'), index=False)
    val_fold.to_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds', f'val_fold_{i}.csv'), index=False)

fold1 = pd.read_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds', 'val_fold_0.csv'))
fold2 = pd.read_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds', 'val_fold_1.csv'))
fold3 = pd.read_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds', 'val_fold_2.csv'))
fold4 = pd.read_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds', 'val_fold_3.csv'))
fold5 = pd.read_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds', 'val_fold_4.csv'))

age1 = fold1['age'].values
age2 = fold2['age'].values
age3 = fold3['age'].values
age4 = fold4['age'].values
age5 = fold5['age'].values

plot_age_curves(age1, age2, age3, age4, age5, '/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds', 'age_curves')

# --- Load existing longitudinal folds ---
longitudinal_train_folds = []
longitudinal_val_folds = []
for i in range(5):
    fold_train = pd.read_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds', f'train_fold_{i}.csv'))
    fold_val = pd.read_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds', f'val_fold_{i}.csv'))
    longitudinal_train_folds.append(fold_train)
    longitudinal_val_folds.append(fold_val)

# --- Load new participants (cross-sectional) ---
participants_cs = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds/participants_cs.csv')
existing_ids = set(pd.concat(longitudinal_val_folds)['participant_id'])
new_participants = participants_cs[~participants_cs['participant_id'].isin(existing_ids)].reset_index(drop=True)
print(len(new_participants))

# --- Split new participants into 5 folds ---
age_values = new_participants['age'].values
binned_age = pd.qcut(age_values, q=5, labels=False)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
folds_cs = list(skf.split(new_participants, binned_age))

# --- Append CS folds to longitudinal folds ---
for i, (train_idx, val_idx) in enumerate(folds_cs):
    train_fold = new_participants.iloc[train_idx].reset_index(drop=True)
    val_fold = new_participants.iloc[val_idx].reset_index(drop=True)
    train_fold = train_fold[['participant_id', 'sex', 'age']]
    val_fold = val_fold[['participant_id', 'sex', 'age']]
    combined_train = pd.concat([longitudinal_train_folds[i], train_fold]).reset_index(drop=True)
    combined_train = combined_train[['participant_id', 'sex', 'age']]
    combined_val = pd.concat([longitudinal_val_folds[i], val_fold]).reset_index(drop=True)
    combined_val = combined_val[['participant_id', 'sex', 'age']]
    combined_train.to_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds', f'cs_train_fold_{i}.csv'), index=False)
    combined_val.to_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds', f'cs_val_fold_{i}.csv'), index=False)

fold1 = pd.read_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds', 'cs_val_fold_0.csv'))
fold2 = pd.read_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds', 'cs_val_fold_1.csv'))
fold3 = pd.read_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds', 'cs_val_fold_2.csv'))
fold4 = pd.read_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds', 'cs_val_fold_3.csv'))
fold5 = pd.read_csv(os.path.join('/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds', 'cs_val_fold_4.csv'))

age1 = fold1['age'].values
age2 = fold2['age'].values
age3 = fold3['age'].values
age4 = fold4['age'].values
age5 = fold5['age'].values

plot_age_curves(age1, age2, age3, age4, age5, '/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds', 'cs_age_curves')




