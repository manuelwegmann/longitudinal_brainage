import numpy
import pandas
from .prep_data import full_data_load
df = full_data_load()
from .data_analysis_tools import plot_mri_info_by_age
a=df['mr_sessions'].unique()
print("These are", a)


