import numpy as np
import pandas as pd

#Consider absolutely basic overview of features with min max and mean +  std_dev before processing anything

#import of data
from .prep_data import load_basic_overview, check_folders_exist, update_ages, add_duration, exclude_single_scan_participants
from .prep_data import split_by_gender
df = load_basic_overview('/mimer/NOBACKUP/groups/brainage/data/oasis3/participants.tsv') #load overview of data
oasis3_fp = '/mimer/NOBACKUP/groups/brainage/data/oasis3'
df = check_folders_exist(df, oasis3_fp) #check if all folders with data exist
df = exclude_single_scan_participants(df) #delete subjects with not enough scans
df = update_ages(df, oasis3_fp) #add ages at first scan
df = add_duration(df) #add duration in days between first and last scan
mf,ff = split_by_gender(df) #get sex-specific dataframes

from .data_analysis_tools import basic_age_analysis, plot_age_histograms
print("Let's look at the basic age statistics for all the subjects:")
print(basic_age_analysis(df))
print("Age analysis for male:")
print(basic_age_analysis(mf))
print("Age analysis for female:")
print(basic_age_analysis(ff))
plot_age_histograms(df,mf,ff)

from .data_analysis_tools import basic_mr_sessions_analysis, plot_mris_by_age
print("Let's see how many scans we have available per subject:")
print(basic_mr_sessions_analysis(df))
plot_mris_by_age(df)


