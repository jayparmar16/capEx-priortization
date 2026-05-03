import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# The Audit Pipeline Component
# Detects demographic disparities and calculates both raw and risk-adjusted selection rates.

def run_audit(scored_csv_path: str):
    df = pd.read_csv(scored_csv_path)

    metrics = {}

    # 1. Raw Selection Rate by Income Quartile
    # The percentage of properties in each income quartile that were selected for funding.
    raw_selection = df.groupby('income_quartile')['funded_baseline'].mean().to_dict()
    metrics['raw_selection_rate_by_income'] = raw_selection

    # 2. Raw Selection Rate by Racial Composition (Minority Pct Thresholds)
    # Group into quartiles of minority percentage
    df['minority_quartile'] = pd.qcut(df['minority_pct'], q=4, labels=['Low', 'Low-Mid', 'Mid-High', 'High'])
    raw_selection_race = df.groupby('minority_quartile', observed=False)['funded_baseline'].mean().to_dict()
    metrics['raw_selection_rate_by_minority_pct'] = raw_selection_race

    # 3. Risk-Adjusted Selection Rate Gap
    # We want to know: is the gap explained by genuine risk differences, or by the scoring logic itself?
    # We'll use a Logistic Regression to model the probability of being funded based ONLY on physical risk.
    # Then we compare the actual funding rate against the predicted (risk-justified) funding rate.

    X = df[['physical_risk_score']]
    y = df['funded_baseline'].astype(int)

    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X, y)

    # Predict the probability of funding if it were purely based on physical risk
    df['risk_justified_prob'] = clf.predict_proba(X)[:, 1]

    # Calculate the gap: Actual Selection Rate - Risk Justified Probability
    # Positive gap = Group receives MORE funding than their physical risk justifies
    # Negative gap = Group receives LESS funding than their physical risk justifies

    risk_adj_income = {}
    for inc in sorted(df['income_quartile'].unique()):
        subset = df[df['income_quartile'] == inc]
        actual_rate = subset['funded_baseline'].mean()
        justified_rate = subset['risk_justified_prob'].mean()
        risk_adj_income[inc] = actual_rate - justified_rate

    metrics['risk_adjusted_gap_by_income'] = risk_adj_income

    risk_adj_race = {}
    for mq in ['Low', 'Low-Mid', 'Mid-High', 'High']:
        subset = df[df['minority_quartile'] == mq]
        actual_rate = subset['funded_baseline'].mean()
        justified_rate = subset['risk_justified_prob'].mean()
        risk_adj_race[mq] = actual_rate - justified_rate

    metrics['risk_adjusted_gap_by_minority'] = risk_adj_race

    return df, metrics

if __name__ == "__main__":
    df, metrics = run_audit("data/portfolio_scored.csv")
    print("--- AUDIT METRICS ---")
    print("\nRaw Selection Rates by Income Quartile (4=High, 1=Low):")
    for k, v in metrics['raw_selection_rate_by_income'].items():
        print(f"  Q{k}: {v:.2%}")

    print("\nRisk-Adjusted Gap by Income Quartile:")
    for k, v in metrics['risk_adjusted_gap_by_income'].items():
        print(f"  Q{k}: {v:+.2%} (Positive means over-funded relative to physical risk)")

    print("\nRaw Selection Rates by Minority Pct Quartile:")
    for k, v in metrics['raw_selection_rate_by_minority_pct'].items():
        print(f"  {k}: {v:.2%}")

    print("\nRisk-Adjusted Gap by Minority Pct Quartile:")
    for k, v in metrics['risk_adjusted_gap_by_minority'].items():
        print(f"  {k}: {v:+.2%} (Positive means over-funded relative to physical risk)")

    df.to_csv("data/portfolio_audited.csv", index=False)
