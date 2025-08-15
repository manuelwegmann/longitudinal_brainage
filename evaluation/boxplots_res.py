import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from plotting import (
    set_r_params,
    get_figures,
    set_style_ax,
    set_size,
    save_figure,
)

"""
Load Data
"""
paths = {
    'CS CNN': '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/CS_CNN/longitudinal_predictions.csv',
    'LILAC': '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC/predictions_all_folds.csv',
    'LILAC+': '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_plus/predictions_all_folds.csv',
    'AEM-4 (age)': '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/AE_age_4/predictions_all_folds.csv',
}

# Load data
results = {name: pd.read_csv(path) for name, path in paths.items()}

selected_models = ['CS CNN', 'LILAC', 'LILAC+', 'AEM-4 (age)']

# Combine residuals and age into a single DataFrame
all_data = []

for model_name in selected_models:
    df = results[model_name].copy()
    df['Residual'] = df['Target'] - df['Prediction']
    df['Model'] = model_name
    all_data.append(df[['Residual', 'Age', 'Model', 'Target']])

combined_df = pd.concat(all_data, ignore_index=True)

# Define age bins
bins = [0, 60, 65, 70, 200]
labels = ['<60 (n=427)', '60-65 (n=350)', '65-70 (n=532)', '70+ (n=730)']
combined_df['Age Group'] = pd.cut(combined_df['Age'], bins=bins, labels=labels, right=False)

set_r_params(small=8)

fig, axes = get_figures(n_rows=2, n_cols=1, figsize=(6, 6), sharex=False, sharey=False)
# Boxplot of residuals by age group
sns.boxplot(
    data=combined_df,
    x='Age Group',
    y='Residual',
    hue='Model',
    ax=axes[0],
    palette='Set2',
    fliersize=1
)

axes[0].set_ylabel('Residual [years]')
axes[0].set_xlabel('Age Group [years]')
axes[0].legend(loc='upper left', ncol=4, title='Model')
axes[0].set_title('Residuals by Age Group')

# Define interval target bins
interval_bins = [0, 1, 3, 5, 100]
interval_labels = ['<1 (n=173)', '1–3 (n=570)', '3–5 (n=591)', '5+ (n=705)']
combined_df['Interval Group'] = pd.cut(combined_df['Target'], bins=interval_bins, labels=interval_labels, right=False)

# Boxplot of residuals by target interval
sns.boxplot(
    data=combined_df,
    x='Interval Group',
    y='Residual',
    hue='Model',
    ax=axes[1],
    palette='Set2',
    fliersize=1
)
axes[1].set_ylabel('Residual [years]')
axes[1].set_xlabel('Target [years]')
axes[1].legend_.remove()
axes[1].set_title('Residuals by Target Group')


# Style axes
fig = set_style_ax(fig, axes, both_axes=False)

# Resize
fig = set_size(fig, 6, 6)

os.makedirs("figures", exist_ok=True)
save_figure(fig, "figures/boxplots_res.png")

# Compute IQR outlier thresholds and counts for Age
outlier_stats = []

grouped = combined_df.groupby(['Model', 'Age Group'])

for (model, age_group), group in grouped:
    q1 = group['Residual'].quantile(0.25)
    q3 = group['Residual'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    mean_residual = group['Residual'].mean()

    # Count outliers
    over = (group['Residual'] > upper_bound).sum()
    under = (group['Residual'] < lower_bound).sum()
    total = over + under
    n = len(group)

    outlier_stats.append({
        'Model': model,
        'Age Group': age_group,
        'Q1': q1,
        'Q3': q3,
        'IQR': iqr,
        'Mean Residual': mean_residual,
        'Underestimating Outliers': under,
        'Overestimating Outliers': over,
        'Total Outliers': total,
        '% Total Outliers': 100 * total / n if n > 0 else np.nan,
        '% Under': 100 * under / total if total > 0 else np.nan,
        '% Over': 100 * over / total if total > 0 else np.nan,
        'N Samples': n
    })

outlier_df = pd.DataFrame(outlier_stats)
print(outlier_df.sort_values(['Age Group', 'Model']))
outlier_df.to_csv('stats/age_residual_analysis.csv', index=False)


# Compute IQR outlier thresholds and counts for Target
interval_outlier_stats = []

grouped_interval = combined_df.groupby(['Model', 'Interval Group'])

for (model, interval_group), group in grouped_interval:
    q1 = group['Residual'].quantile(0.25)
    q3 = group['Residual'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    mean_residual = group['Residual'].mean()

    # Count outliers
    over = (group['Residual'] > upper_bound).sum()
    under = (group['Residual'] < lower_bound).sum()
    total = over + under
    n = len(group)

    interval_outlier_stats.append({
        'Model': model,
        'Target Interval Group': interval_group,
        'Q1': q1,
        'Q3': q3,
        'IQR': iqr,
        'Mean Residual': mean_residual,
        'Underestimating Outliers': under,
        'Overestimating Outliers': over,
        'Total Outliers': total,
        '% Total Outliers': 100 * total / n if n > 0 else np.nan,
        '% Under': 100 * under / total if total > 0 else np.nan,
        '% Over': 100 * over / total if total > 0 else np.nan,
        'N Samples': n
    })

interval_outlier_df = pd.DataFrame(interval_outlier_stats)
print(interval_outlier_df.sort_values(['Target Interval Group', 'Model']))
interval_outlier_df.to_csv('stats/target_residual_analysis.csv', index=False)

