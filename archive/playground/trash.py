import pandas as pd

results = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/old_CI_CN_model_comparison.csv')
cs_1 = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/CS_CI_final/predictions_ci.csv')
cs_2 = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/CS_CI_final/predictions_cn.csv')


# Combine the prediction datasets
preds = pd.concat([cs_1, cs_2], ignore_index=True)
preds = preds.rename(columns={'Prediction': 'Predicted Baseline Age'})
preds = preds.rename(columns={'Sessions': 'Session 1'})

# Make sure column names match:
# e.g., predictions file should have 'Prediction', 'Participant_ID', 'Session'
#print(preds.columns)
#print(results.columns)

# Merge prediction into results
results_with_baseline = results.merge(
    preds[['Participant_ID', 'Session 1', 'Predicted Baseline Age']],
    on=['Participant_ID', 'Session 1'],
    how='left'
)

# Rename column
print(results_with_baseline)

# Save if needed
results_with_baseline.to_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/CI_CN_model_comparison.csv', index=False)