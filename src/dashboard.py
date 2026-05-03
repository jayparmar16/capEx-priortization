import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.abspath('src'))
from reranker import parity_constrained_rerank

st.set_page_config(layout="wide", page_title="AI Governance: CRE Allocation")

st.title("Florida CRE Portfolio Prioritization Dashboard")
st.markdown("### AI Governance and Fairness Layer")

@st.cache_data
def load_data():
    # Load the audited dataset
    if os.path.exists("data/portfolio_audited.csv"):
        return pd.read_csv("data/portfolio_audited.csv")
    else:
        st.error("Data not found. Please run the pipeline first.")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    st.sidebar.header("Asset Manager Controls")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Parity Guardrail Settings")
    apply_guardrail = st.sidebar.checkbox("Enable Parity Re-ranker", value=False)
    tolerance = st.sidebar.slider("Parity Tolerance (± %)", min_value=1, max_value=20, value=10, step=1) / 100.0

    target_k = int(df['funded_baseline'].sum())
    overall_rate = target_k / len(df)

    if apply_guardrail:
        df_display, tax = parity_constrained_rerank(df, target_k, tolerance=tolerance)
        funded_col = 'funded_reranked'
    else:
        df_display = df.copy()
        funded_col = 'funded_baseline'
        tax = None

    # Top level metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Portfolio Properties", len(df))
    col2.metric("Properties Funded (Top-K)", target_k)
    col3.metric("Overall Selection Rate", f"{overall_rate:.1%}")

    st.markdown("---")

    # Dashboard layout: 2 columns
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("Selection Rate by Income Quartile")

        # Calculate rates
        rates = df_display.groupby('income_quartile')[funded_col].mean().reset_index()
        rates['income_quartile'] = rates['income_quartile'].astype(int)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=rates, x='income_quartile', y=funded_col, ax=ax, palette="Blues")
        ax.axhline(overall_rate, ls='--', color='red', label=f'Overall Average ({overall_rate:.1%})')
        if apply_guardrail:
            ax.axhline(overall_rate + tolerance, ls=':', color='orange', label='Tolerance Upper')
            ax.axhline(overall_rate - tolerance, ls=':', color='orange', label='Tolerance Lower')

        ax.set_ylabel("Selection Rate")
        ax.set_xlabel("Income Quartile (1=Low, 4=High)")
        ax.set_ylim(0, 1)
        ax.legend()
        st.pyplot(fig)

    with right_col:
        st.subheader("Trust Intervention: Risk-Adjusted Audit")
        st.markdown("This metric isolates whether the gap in funding is explained by *genuine physical risk* (wind/flood) or by the financial scoring logic itself.")

        # Show the risk-adjusted gap (from baseline, even if reranked, to show the *why*)
        gap_data = []
        for inc in sorted(df['income_quartile'].unique()):
            subset = df[df['income_quartile'] == inc]
            actual_rate = subset['funded_baseline'].mean()
            justified_rate = subset['risk_justified_prob'].mean()
            gap = actual_rate - justified_rate
            gap_data.append({'Income Quartile': int(inc), 'Unexplained Gap': gap})

        gap_df = pd.DataFrame(gap_data)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        colors = ['red' if x < 0 else 'green' for x in gap_df['Unexplained Gap']]
        sns.barplot(data=gap_df, x='Income Quartile', y='Unexplained Gap', ax=ax2, palette=colors)
        ax2.axhline(0, color='black', lw=1)
        ax2.set_ylabel("Gap vs Risk-Justified Rate")
        st.pyplot(fig2)

        if gap_df['Unexplained Gap'].iloc[0] < -0.1:
            st.error("⚠️ **Unjustified Disparity Detected**: Lower-income quartiles are significantly under-funded relative to their physical risk. This is likely driven by the financial logic (NOI/Value) over-weighting wealthy coastal properties.")
        else:
            st.success("✅ Allocation appears aligned with physical risk.")

    if apply_guardrail and tax is not None:
        st.markdown("---")
        st.subheader("Guardrail Impact & Trust Tax")
        tcol1, tcol2 = st.columns(2)
        tcol1.warning(f"**Properties Swapped:** {tax['swaps_count']}\n\nThis is the number of high-risk/high-value properties bumped to achieve demographic parity.")
        tcol2.warning(f"**Precision Drop:** {tax['precision_drop']:.1%}\n\nThe drop in top-K overlap compared to the baseline risk-weighted ranking.")

    st.markdown("---")
    st.subheader("Geographic View of Funded Properties")

    # Map
    m = folium.Map(location=[26.5, -81.0], zoom_start=7)

    for idx, row in df_display.iterrows():
        is_funded = row[funded_col]
        color = 'green' if is_funded else 'gray'
        radius = 5 if is_funded else 2

        popup_text = f"Value: ${row['property_value']:,.0f}<br>Income Q: {row['income_quartile']}<br>Risk Score: {row['physical_risk_score']:.2f}"

        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=popup_text
        ).add_to(m)

    st_folium(m, width=1200, height=400)

    st.markdown("---")
    st.subheader("Funded Portfolio Details")
    display_cols = ['name', 'county', 'income_quartile', 'flood_zone', 'physical_risk_score', 'financial_exposure_score', 'property_value']
    st.dataframe(df_display[df_display[funded_col]][display_cols].sort_values('physical_risk_score', ascending=False))
