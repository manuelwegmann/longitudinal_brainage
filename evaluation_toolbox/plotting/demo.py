import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd


results_LILAC = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_plus_final_run/predictions_all_folds.csv')

age_LILAC = results_LILAC['Age']
targets_LILAC = results_LILAC['Target']
preds_LILAC = results_LILAC['Prediction']
res_LILAC = targets_LILAC - preds_LILAC
idx_M = results_LILAC[results_LILAC['Sex (M)'] == 1].index
idx_F = results_LILAC[results_LILAC['Sex (M)'] == 0].index

# Create Q-Q plot
stats.probplot(res_LILAC, dist="norm", plot=plt)
plt.title("Q-Q plot of residuals")

# Save the plot to disk
plt.savefig('qqplot_residuals_LILAC.png', dpi=300, bbox_inches='tight')

plt.show()

stat, p = stats.shapiro(res_LILAC)
print(f"Shapiro-Wilk test p-value: {p:.4f}")

if p > 0.05:
    print("Residuals are likely normal (fail to reject H0)")
else:
    print("Residuals are likely NOT normal (reject H0)")

stat, p = stats.mannwhitneyu(res_LILAC[idx_M], res_LILAC[idx_F], alternative='two-sided')
print(f"U-statistic: {stat:.3f}, p-value: {p:.4f}")