import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# The Audit Pipeline Component
# Detects demographic disparities and calculates raw and risk-adjusted selection rates
# using THREE complementary baselines so the audit signal is robust to a single
# method's hyperparameter choices:
#   (a) Balanced logistic regression  -> risk_justified_prob_balanced  (legacy)
#   (b) Plain logistic regression     -> risk_justified_prob_plain
#   (c) Counterfactual top-K ranking  -> risk_only_funded
#
# (c) is the most direct and the easiest to explain to a non-statistician:
# "If we ranked properties by physical risk alone and funded the top K, who would
#  get funded?" Then we compare that group selection rate to the AI's actual rate.


def _fit_logreg(df, balanced):
    X = df[["physical_risk_score"]]
    y = df["funded_baseline"].astype(int)
    kwargs = {"class_weight": "balanced"} if balanced else {}
    clf = LogisticRegression(**kwargs)
    clf.fit(X, y)
    return clf.predict_proba(X)[:, 1]


def _counterfactual_topk(df, target_k):
    """Top-K by physical_risk_score only. Uses 'first' to break ties deterministically."""
    risk_rank = df["physical_risk_score"].rank(ascending=False, method="first")
    return (risk_rank <= target_k).astype(bool)


def _group_gap(df, group_col, actual_col, baseline_col, group_order=None):
    """For each group, return (actual_rate - baseline_rate)."""
    out = {}
    keys = group_order if group_order is not None else sorted(df[group_col].dropna().unique())
    for k in keys:
        sub = df[df[group_col] == k]
        if len(sub) == 0:
            continue
        actual_rate = sub[actual_col].mean()
        baseline_rate = sub[baseline_col].mean()
        out[k] = actual_rate - baseline_rate
    return out


def run_audit(scored_csv_path: str):
    df = pd.read_csv(scored_csv_path)

    metrics = {}
    target_k = int(df["funded_baseline"].sum())

    # 1. Raw selection rates
    metrics["raw_selection_rate_by_income"] = (
        df.groupby("income_quartile")["funded_baseline"].mean().to_dict()
    )

    df["minority_quartile"] = pd.qcut(
        df["minority_pct"], q=4, labels=["Low", "Low-Mid", "Mid-High", "High"]
    )
    metrics["raw_selection_rate_by_minority_pct"] = (
        df.groupby("minority_quartile", observed=False)["funded_baseline"].mean().to_dict()
    )

    # 2. Three risk-justified baselines
    df["risk_justified_prob_balanced"] = _fit_logreg(df, balanced=True)
    df["risk_justified_prob_plain"] = _fit_logreg(df, balanced=False)
    df["risk_only_funded"] = _counterfactual_topk(df, target_k)

    # Keep legacy column name so the dashboard and existing notebooks keep working.
    df["risk_justified_prob"] = df["risk_justified_prob_balanced"]

    # 3. Per-group gaps under each baseline
    income_order = sorted(df["income_quartile"].dropna().unique())
    minority_order = ["Low", "Low-Mid", "Mid-High", "High"]

    metrics["risk_adjusted_gap_by_income"] = _group_gap(
        df, "income_quartile", "funded_baseline", "risk_justified_prob_balanced", income_order
    )
    metrics["risk_adjusted_gap_by_income_plain"] = _group_gap(
        df, "income_quartile", "funded_baseline", "risk_justified_prob_plain", income_order
    )
    metrics["risk_adjusted_gap_by_income_counterfactual"] = _group_gap(
        df, "income_quartile", "funded_baseline", "risk_only_funded", income_order
    )

    metrics["risk_adjusted_gap_by_minority"] = _group_gap(
        df, "minority_quartile", "funded_baseline", "risk_justified_prob_balanced", minority_order
    )
    metrics["risk_adjusted_gap_by_minority_counterfactual"] = _group_gap(
        df, "minority_quartile", "funded_baseline", "risk_only_funded", minority_order
    )

    return df, metrics


def _print_dict(name, d, fmt="{:+.2%}"):
    print(f"\n{name}:")
    for k, v in d.items():
        print(f"  {k}: {fmt.format(v)}")


if __name__ == "__main__":
    df, metrics = run_audit("data/portfolio_scored.csv")

    print("=" * 60)
    print("AUDIT METRICS")
    print("=" * 60)

    _print_dict("Raw selection rate by income quartile", metrics["raw_selection_rate_by_income"], "{:.2%}")
    _print_dict("Raw selection rate by minority pct quartile", metrics["raw_selection_rate_by_minority_pct"], "{:.2%}")

    print("\n" + "-" * 60)
    print("RISK-ADJUSTED GAP BY INCOME QUARTILE — three methods")
    print("(actual selection rate - risk-justified rate)")
    print("Negative means under-funded relative to physical risk.")
    print("-" * 60)
    _print_dict("(a) Balanced logistic regression [legacy]", metrics["risk_adjusted_gap_by_income"])
    _print_dict("(b) Plain logistic regression",             metrics["risk_adjusted_gap_by_income_plain"])
    _print_dict("(c) Counterfactual top-K ranking",          metrics["risk_adjusted_gap_by_income_counterfactual"])

    _print_dict("\nRisk-adjusted gap by minority quartile (counterfactual)",
                metrics["risk_adjusted_gap_by_minority_counterfactual"])

    df.to_csv("data/portfolio_audited.csv", index=False)
    print("\nWrote data/portfolio_audited.csv")
