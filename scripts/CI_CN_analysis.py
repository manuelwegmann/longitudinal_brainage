import pandas as pd
import numpy as np
import os

ci_results_CS = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/CS_CI_final/longitudinal_predictions_ci.csv')
cn_results_CS = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/CS_CI_final/longitudinal_predictions_cn.csv')

ci_results_LILAC = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_CI_final/predictions_CI.csv')
cn_results_LILAC = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_CI_final/predictions_CN.csv')

ci_results_LILACp = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILACp_CI_final/predictions_CI.csv')
cn_results_LILACp = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILACp_CI_final/predictions_CN.csv')

for df in [
    ci_results_CS, cn_results_CS,
    ci_results_LILAC, cn_results_LILAC,
    ci_results_LILACp, cn_results_LILACp
]:
    df["Target_round2"] = df["Target"].round(2)

def check_match(df1, df2, df3, label):
    print(f"\n--- Checking {label} ---")
    
    # Compare Participant_ID
    ids1, ids2, ids3 = set(df1["Participant_ID"]), set(df2["Participant_ID"]), set(df3["Participant_ID"])
    
    print("Participant_ID match across all models:", ids1 == ids2 == ids3)
    if not (ids1 == ids2 == ids3):
        print(" - In CS not in LILAC:", ids1 - ids2)
        print(" - In LILAC not in CS:", ids2 - ids1)
        print(" - In LILACp not in CS:", ids3 - ids1)

    # Compare Target
    t1, t2, t3 = set(df1["Target_round2"]), set(df2["Target_round2"]), set(df3["Target_round2"])
    
    print("Target match across all models:", t1 == t2 == t3)
    if not (t1 == t2 == t3):
        print(" - In CS not in LILAC:", t1 - t2)
        print(" - In LILAC not in CS:", t2 - t1)
        print(" - In LILACp not in CS:", t3 - t1)


# Run checks
check_match(ci_results_CS, ci_results_LILAC, ci_results_LILACp, "CI")
check_match(cn_results_CS, cn_results_LILAC, cn_results_LILACp, "CN")

import pandas as pd

# -------------------------
# 1. Load all CI/CN results
# -------------------------

ci_results_CS = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/CS_CI_final/longitudinal_predictions_ci.csv')
cn_results_CS = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/CS_CI_final/longitudinal_predictions_cn.csv')

ci_results_LILAC = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_CI_final/predictions_CI.csv')
cn_results_LILAC = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILAC_CI_final/predictions_CN.csv')

ci_results_LILACp = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILACp_CI_final/predictions_CI.csv')
cn_results_LILACp = pd.read_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/LILACp_CI_final/predictions_CN.csv')


def merge_group(cs_df, lilac_df, lilacp_df, group_label):
    
    # Rename pace columns
    cs_df = cs_df.rename(columns={"Pace": "Pace_CS"})
    lilac_df = lilac_df.rename(columns={"Pace": "Pace_LILAC"})
    lilacp_df = lilacp_df.rename(columns={"Pace": "Pace_LILACp"})
    
    # Keep metadata from CS file + its pace
    base = cs_df[[
        "Participant_ID", 
        "Session 1", 
        "Session 2",
        "Age", 
        "Sex (M)",
        "Pace_CS"
    ]]
    
    # Merge pace from other models
    merged = (
        base
        .merge(lilac_df[["Participant_ID", "Pace_LILAC"]], on="Participant_ID")
        .merge(lilacp_df[["Participant_ID", "Pace_LILACp"]], on="Participant_ID")
    )
    
    # Add group label (CI / CN)
    merged["Group"] = group_label
    
    return merged


merged_CI = merge_group(ci_results_CS, ci_results_LILAC, ci_results_LILACp, "CI")
merged_CN = merge_group(cn_results_CS, cn_results_LILAC, cn_results_LILACp, "CN")


final_df = pd.concat([merged_CI, merged_CN], ignore_index=True)

final_df.to_csv('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/CI_CN_model_comparison.csv', index=False)

import scipy.stats as stats

# Extract pace values for CI and CN
ci_cs = merged_CI["Pace_CS"].values
cn_cs = merged_CN["Pace_CS"].values

ci_lilac = merged_CI["Pace_LILAC"].values
cn_lilac = merged_CN["Pace_LILAC"].values

ci_lilacp = merged_CI["Pace_LILACp"].values
cn_lilacp = merged_CN["Pace_LILACp"].values


def one_sided_mwu(ci, cn, model_name):
    # Alternative: CI > CN
    u, p = stats.mannwhitneyu(ci, cn, alternative="greater")
    print(f"{model_name}: U = {u:.1f}, one-sided p = {p:.5f}")


print("\n--- One-sided Mann-Whitney U tests (CI > CN) ---\n")
one_sided_mwu(ci_cs, cn_cs, "CS")
one_sided_mwu(ci_lilac, cn_lilac, "LILAC")
one_sided_mwu(ci_lilacp, cn_lilacp, "LILACp")