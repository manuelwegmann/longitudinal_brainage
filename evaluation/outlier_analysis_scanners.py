import os
import numpy as np
import pandas as pd
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--predictions_file', default='blank', type=str, help="directory of the results to be evaluated")

    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.predictions_file)
    df['error'] = np.abs(df['Target'].values - df['Prediction'].values)

    threshold = 3
    outliers = df[df['error'] > threshold]

    """General Scanner Analysis"""
    scanner_pair_list = []
    same_scanner_count = 0
    different_scanner_count = 0
    for _,row in df.iterrows():
        scanner_1 = row['Scanner 1']
        scanner_2 = row['Scanner 2']
        if scanner_1 == scanner_2:
            same_scanner_count += 1
        else:
            different_scanner_count += 1
        pair = f"{scanner_1}_{scanner_2}"
        scanner_pair_list.append(pair)
    unique, counts = np.unique(scanner_pair_list, return_counts=True)

    print(unique)  # ['apple' 'banana' 'orange']
    print(counts)  # [3 2 1]

    print(same_scanner_count)
    print(different_scanner_count)


        # Add scanner pair column to both full and outlier DataFrames
    df['Scanner Pair'] = df.apply(lambda row: f"{row['Scanner 1']}_{row['Scanner 2']}", axis=1)
    outliers['Scanner Pair'] = outliers.apply(lambda row: f"{row['Scanner 1']}_{row['Scanner 2']}", axis=1)

    # Get value counts for all pairs
    total_counts = df['Scanner Pair'].value_counts()
    outlier_counts = outliers['Scanner Pair'].value_counts()

    # Combine into a DataFrame for comparison
    scanner_analysis = pd.DataFrame({
        'Total': total_counts,
        'Outliers': outlier_counts
    }).fillna(0)

    # Calculate percentage of outliers for each scanner pair
    scanner_analysis['Outlier Rate'] = scanner_analysis['Outliers'] / scanner_analysis['Total']

    # Sort by outlier rate descending
    scanner_analysis = scanner_analysis.sort_values(by='Outlier Rate', ascending=False)

    print("\n📊 Scanner pair overrepresentation in outliers:")
    print(scanner_analysis)

    from scipy.stats import chi2_contingency
    # Optional: save to CSV
    # scanner_analysis.to_csv("scanner_pair_outlier_analysis.csv")
    # Create binary outlier column
    threshold = 3
    df['is_outlier'] = df['error'] > threshold

    # Create scanner pair column
    df['Scanner Pair'] = df.apply(lambda row: f"{row['Scanner 1']}_{row['Scanner 2']}", axis=1)

    # Create contingency table
    contingency = pd.crosstab(df['Scanner Pair'], df['is_outlier'])

    # Run chi-squared test
    chi2, p, dof, expected = chi2_contingency(contingency)

    print(f"Chi-squared statistic: {chi2:.3f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p:.5f}")

    if p < 0.05:
        print("✅ Scanner pair has a statistically significant effect on the chance of being an outlier.")
    else:
        print("⚠️ No significant association found between scanner pair and outlier status.")

    import pandas as pd
    import statsmodels.formula.api as smf

    # Binary outlier variable
    df['is_outlier'] = (df['error'] > threshold).astype(int)

    # Make sure categorical variables are treated properly
    df['Scanner_Pair'] = df.apply(lambda row: f"{row['Scanner 1']}_{row['Scanner 2']}", axis=1)
    df['Sex'] = df['Sex (M)']

    # Example formula
    model = smf.logit("is_outlier ~ C(Scanner_Pair) + Age + Sex + Target", data=df).fit()

    print(model.summary())

    # Define a function to create an unordered scanner pair string
    def make_unordered_pair(row):
        scanners = sorted([row['Scanner 1'], row['Scanner 2']])
        return f"{scanners[0]}_{scanners[1]}"

    # Apply it to create the 'Scanner_Pair' column
    df['Scanner_Pair'] = df.apply(make_unordered_pair, axis=1)

    # Count occurrences of each scanner pair
    pair_counts = df['Scanner_Pair'].value_counts()

    # Threshold for minimum number of samples
    min_samples = 50

    # Get scanner pairs to keep
    frequent_pairs = pair_counts[pair_counts >= min_samples].index

    # Create new column, keep frequent scanner pairs, label others as 'Other'
    df['Scanner_Pair_Simplified'] = df['Scanner_Pair'].apply(
        lambda x: x if x in frequent_pairs else 'Other'
    )

    # Check counts to verify
    print(df['Scanner_Pair_Simplified'].value_counts())

    # Then use this in your logistic regression:
    model = smf.logit("is_outlier ~ C(Scanner_Pair_Simplified) + Age + Sex + Target", data=df).fit(maxiter=100)

    print(model.summary())
    

        
