"""Generates the chart PNGs embedded in report2.html from the live pipeline output."""
import os
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from reranker import parity_constrained_rerank

OUT_DIR = os.path.dirname(__file__)
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "portfolio_audited.csv")

sns.set_style("whitegrid")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

df = pd.read_csv(DATA)
target_k = int(df["funded_baseline"].sum())
overall_rate = target_k / len(df)
df_rerank, tax = parity_constrained_rerank(df, target_k, tolerance=0.10)


# 1) Baseline selection rate by income quartile
rates = df.groupby("income_quartile")["funded_baseline"].mean().reset_index()
rates["income_quartile"] = rates["income_quartile"].astype(int)
fig, ax = plt.subplots(figsize=(7, 4.2))
sns.barplot(data=rates, x="income_quartile", y="funded_baseline",
            ax=ax, palette="Blues", edgecolor="#1f3a5f")
ax.axhline(overall_rate, ls="--", color="#e74c3c",
           label=f"Portfolio average ({overall_rate:.0%})")
ax.set_ylim(0, 0.6)
ax.set_xlabel("Income Quartile (1 = lowest, 4 = highest)")
ax.set_ylabel("Selection rate (share of properties funded)")
ax.set_title("Baseline AI: Funding share by neighborhood income")
ax.legend(loc="upper left", frameon=True)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.0%}", (p.get_x() + p.get_width() / 2, p.get_height() + 0.015),
                ha="center", fontsize=10, color="#1f3a5f")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_baseline_selection.png"), dpi=140)
plt.close()


# 2) Risk-adjusted gap by income quartile
gap_rows = []
for inc in sorted(df["income_quartile"].unique()):
    sub = df[df["income_quartile"] == inc]
    gap_rows.append({
        "income_quartile": int(inc),
        "gap": sub["funded_baseline"].mean() - sub["risk_justified_prob"].mean()
    })
gap_df = pd.DataFrame(gap_rows)
fig, ax = plt.subplots(figsize=(7, 4.2))
colors = ["#ef4444" if v < 0 else "#10b981" for v in gap_df["gap"]]
sns.barplot(data=gap_df, x="income_quartile", y="gap",
            ax=ax, palette=colors, edgecolor="#1f3a5f")
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("Income Quartile (1 = lowest, 4 = highest)")
ax.set_ylabel("Actual funding − risk-justified funding")
ax.set_title("Risk-Adjusted Gap: where the AI deviates from physical risk")
for p, v in zip(ax.patches, gap_df["gap"]):
    offset = 0.012 if v >= 0 else -0.025
    ax.annotate(f"{v:+.0%}", (p.get_x() + p.get_width() / 2, p.get_height() + offset),
                ha="center", fontsize=10, color="#1f3a5f")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_risk_adjusted_gap.png"), dpi=140)
plt.close()


# 2b) Three-method robustness comparison
methods = []
for inc in sorted(df["income_quartile"].dropna().unique()):
    sub = df[df["income_quartile"] == inc]
    actual = sub["funded_baseline"].mean()
    methods.append({
        "Quartile": int(inc),
        "Balanced regression": actual - sub["risk_justified_prob_balanced"].mean(),
        "Plain regression":    actual - sub["risk_justified_prob_plain"].mean(),
        "Counterfactual top-K": actual - sub["risk_only_funded"].mean(),
    })
methods_df = pd.DataFrame(methods)
melted = methods_df.melt(id_vars="Quartile", var_name="Method", value_name="Gap")
fig, ax = plt.subplots(figsize=(8.2, 4.6))
sns.barplot(
    data=melted, x="Quartile", y="Gap", hue="Method",
    ax=ax, palette=["#94a3b8", "#3b82f6", "#10b981"], edgecolor="#1f3a5f",
)
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("Income Quartile (1 = lowest, 4 = highest)")
ax.set_ylabel("Actual − risk-justified selection rate")
ax.set_title("Three baselines, three different verdicts on the same AI")
ax.legend(loc="lower right", frameon=True, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02b_three_method_comparison.png"), dpi=140)
plt.close()


# 3) Before/after the parity re-ranker
before = df.groupby("income_quartile")["funded_baseline"].mean()
after = df_rerank.groupby("income_quartile")["funded_reranked"].mean()
combined = pd.DataFrame({
    "Baseline AI": before,
    "After Fairness Guardrail": after,
}).reset_index()
combined["income_quartile"] = combined["income_quartile"].astype(int)
melted = combined.melt(id_vars="income_quartile", var_name="System", value_name="Selection rate")
fig, ax = plt.subplots(figsize=(7.5, 4.4))
sns.barplot(data=melted, x="income_quartile", y="Selection rate", hue="System",
            ax=ax, palette=["#94a3b8", "#3b82f6"], edgecolor="#1f3a5f")
ax.axhline(overall_rate, ls="--", color="#e74c3c",
           label=f"Portfolio average ({overall_rate:.0%})")
ax.axhline(overall_rate + 0.10, ls=":", color="#f59e0b", alpha=0.6, label="±10% tolerance")
ax.axhline(overall_rate - 0.10, ls=":", color="#f59e0b", alpha=0.6)
ax.set_ylim(0, 0.6)
ax.set_xlabel("Income Quartile (1 = lowest, 4 = highest)")
ax.set_ylabel("Selection rate")
ax.set_title("Before vs. after the Parity-Constrained Re-ranker")
ax.legend(loc="upper left", frameon=True, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_before_after_rerank.png"), dpi=140)
plt.close()


# 4) Map-style scatter: lat/lon, funded vs. unfunded after rerank
fig, ax = plt.subplots(figsize=(7.5, 5))
funded = df_rerank[df_rerank["funded_reranked"]]
unfunded = df_rerank[~df_rerank["funded_reranked"]]
ax.scatter(unfunded["lon"], unfunded["lat"], c="#cbd5e1", s=22,
           label="Not funded", alpha=0.85, edgecolor="white", linewidth=0.4)
ax.scatter(funded["lon"], funded["lat"], c="#10b981", s=55,
           label="Funded (after Guardrail)", alpha=0.95, edgecolor="#064e3b", linewidth=0.6)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Geographic distribution of the funded portfolio")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_geographic_view.png"), dpi=140)
plt.close()


# 5) Trust tax summary card
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.axis("off")
swap_count = tax["swaps_count"]
precision_drop = tax["precision_drop"] * 100
ax.text(0.02, 0.78, "The Trust Tax", fontsize=18, weight="bold", color="#1e293b")
ax.text(0.02, 0.55, "What enforcing fairness costs the asset manager",
        fontsize=11, color="#475569")
ax.add_patch(plt.Rectangle((0.02, 0.05), 0.45, 0.40, color="#fef3c7", ec="#f59e0b", lw=1.4))
ax.text(0.245, 0.30, f"{swap_count}", fontsize=34, weight="bold",
        color="#92400e", ha="center")
ax.text(0.245, 0.12, "high-value properties\nswapped out", fontsize=10,
        color="#78350f", ha="center")
ax.add_patch(plt.Rectangle((0.53, 0.05), 0.45, 0.40, color="#fee2e2", ec="#ef4444", lw=1.4))
ax.text(0.755, 0.30, f"{precision_drop:.1f}%", fontsize=34, weight="bold",
        color="#991b1b", ha="center")
ax.text(0.755, 0.12, "drop in financial-optimality\nvs. baseline", fontsize=10,
        color="#7f1d1d", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_trust_tax.png"), dpi=140)
plt.close()


# Print numbers we will reference in the report
print(f"Portfolio size: {len(df)}")
print(f"Funded count: {target_k}")
print(f"Overall selection rate: {overall_rate:.4f}")
print("Baseline rates by quartile:")
print(before)
print("Re-ranked rates by quartile:")
print(after)
print(f"Swaps: {tax['swaps_count']}")
print(f"Precision drop: {tax['precision_drop']:.4f}")
print("Risk-adjusted gaps:")
print(gap_df)
