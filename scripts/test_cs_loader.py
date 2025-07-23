from loader_CS import loader3D
import pandas as pd
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_directory', default='/mimer/NOBACKUP/groups/brainage/data/oasis3', type=str, help="directory of the data (OASIS3)")
    parser.add_argument('--project_data_dir', default ='/mimer/NOBACKUP/groups/brainage/thesis_brainage/data', type=str, help="directory with the updated session files")
    parser.add_argument('--folds_dir', default = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/folds', type = str, help = 'path to participants csv.')

    #data preprocessing arguments
    parser.add_argument('--image_size', nargs=3, type=int, default=[128, 128, 128], help='Input image size as three integers (e.g. 128 128 128)')
    parser.add_argument('--image_channel', default=1, type=int, help="number of channels in the input image")

    #target and optional meta data arguments
    parser.add_argument('--target_name', default='age', type=str, help="name of the target variable")
    parser.add_argument('--optional_meta', nargs='+', default=['sex_M'], help="List of optional meta to be used in the model")
    
    #model architecture arguments
    parser.add_argument('--n_of_blocks', default=4, type=int, help="number of blocks in the encoder")
    parser.add_argument('--initial_channel', default=16, type=int, help="initial channel size after first conv")
    parser.add_argument('--kernel_size', default=3, type=int, help="kernel size")

    #training arguments
    parser.add_argument('--dropout', default=0.1, type=float, help="dropout rate")
    parser.add_argument('--epoch_weight_decay', default=15, type=int, help="epoch after which to decay the learning rate")
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batchsize', default=16, type=int)
    parser.add_argument('--max_epoch', default=20, type=int, help="max epoch")
    parser.add_argument('--epoch', default=0, type=int, help="starting epoch")
    parser.add_argument('--ignore_folds', nargs='+', default = [], help="list of folds to ignore, e.g. 0 1 2")

    parser.add_argument('--folds', default=5, type=int, help = "number of folds for k-fold cv. 0 for no cv.")
    parser.add_argument('--output_directory', default='/mimer/NOBACKUP/groups/brainage/thesis_brainage/results', type=str, help="directory path for saving model and outputs")
    parser.add_argument('--run_name', default='test_run', type=str, help="name of the run")


    args = parser.parse_args()

    return args

args = parse_args()


df = pd.read_csv(f"{args.data_directory}/participants.tsv", sep='\t')
df_train = pd.read_csv(f"{args.folds_dir}/cs_train_fold_0.csv")
ids = df_train['participant_id'].unique()

mr_sessions = 0
for _, row in df.iterrows():
    if row['participant_id'] in ids:
        mr_sessions += row['mr_sessions']

print(len(df_train))
print(mr_sessions / len(df_train))

df_val = pd.read_csv(f"{args.folds_dir}/cs_val_fold_0.csv")
train_data = loader3D(args, df_train).demo
train_data.to_csv('loaderdatacs.csv')
