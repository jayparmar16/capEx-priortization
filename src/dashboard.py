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

# --- Title and Executive Summary ---
st.title("Florida CRE Portfolio Prioritization Dashboard")
st.markdown("""
### AI Governance and Fairness Layer
Asset managers use AI to decide which properties receive capital for hurricane and flood hardening. However, **because property values and income levels are spatially correlated with coastal risk**, purely financial algorithms often create unintended demographic disparities—funding wealthy areas while leaving lower-income areas exposed.

This dashboard acts as an **Audit and Intervention Layer**. It allows you to:
1. **Audit the AI:** See the demographic breakdown of the AI's "fund-first" list and determine if disparities are justified by genuine physical risk.
2. **Apply Guardrails:** Intervene and enforce demographic parity.
3. **Measure the Trust Tax:** See exactly what it costs (in terms of risk/financial optimization) to enforce that parity.
""")
st.markdown("---")

@st.cache_data
def load_data():
    if os.path.exists("data/portfolio_audited.csv"):
        return pd.read_csv("data/portfolio_audited.csv")
    else:
        st.error("Data not found. Please run the pipeline first.")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    # --- Sidebar Controls ---
    st.sidebar.header("Asset Manager Controls")
    st.sidebar.markdown("""
    Use these controls to override the baseline AI recommendation and enforce demographic parity across income quartiles.
    """)
    st.sidebar.markdown("---")

    apply_guardrail = st.sidebar.toggle("🛠️ Enable Parity Re-ranker", value=False, help="Swaps properties to ensure all income quartiles receive a fair share of funding.")
    tolerance = st.sidebar.slider("Parity Tolerance (± %)", min_value=1, max_value=20, value=10, step=1, help="How close to the average selection rate must each group be?") / 100.0

    target_k = int(df['funded_baseline'].sum())
    overall_rate = target_k / len(df)

    if apply_guardrail:
        df_display, tax = parity_constrained_rerank(df, target_k, tolerance=tolerance)
        funded_col = 'funded_reranked'
        st.sidebar.success("Guardrail Active: The list below has been adjusted.")
    else:
        df_display = df.copy()
        funded_col = 'funded_baseline'
        tax = None
        st.sidebar.info("Showing Baseline AI Recommendation (No Guardrails).")

    # --- Top Level Metrics ---
    st.subheader("1. Portfolio Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Portfolio Properties", len(df), help="Total number of commercial properties analyzed.")
    col2.metric("Properties Funded (Top-K)", target_k, help="The number of properties the budget allows us to harden.")
    col3.metric("Overall Selection Rate", f"{overall_rate:.1%}", help="The baseline percentage of the portfolio receiving funding.")

    st.markdown("---")

    # --- Audit Section ---
    st.subheader("2. AI Audit: Demographics & Risk")
    st.markdown("Are we systematically leaving certain communities behind? If so, is it because of genuine weather risk, or is the AI biased toward high property values?")

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown("**Selection Rate by Income Quartile**")
        st.markdown("*Shows what percentage of properties in each income bracket were selected for funding.*")

        rates = df_display.groupby('income_quartile')[funded_col].mean().reset_index()
        rates['income_quartile'] = rates['income_quartile'].astype(int)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=rates, x='income_quartile', y=funded_col, ax=ax, palette="Blues")
        ax.axhline(overall_rate, ls='--', color='red', label=f'Overall Average ({overall_rate:.1%})')
        if apply_guardrail:
            ax.axhline(overall_rate + tolerance, ls=':', color='orange', label='Tolerance Upper')
            ax.axhline(overall_rate - tolerance, ls=':', color='orange', label='Tolerance Lower')

        ax.set_ylabel("Selection Rate")
        ax.set_xlabel("Income Quartile (1=Lowest Income, 4=Highest Income)")
        ax.set_ylim(0, 1)
        ax.legend()
        st.pyplot(fig)

    with right_col:
        st.markdown("**The Risk-Adjusted Gap (The 'Why')**")
        st.markdown("*Isolates whether the gap in funding is explained by genuine physical risk (wind/flood) or by the financial scoring logic itself. A negative bar means the group is under-funded relative to their actual storm risk.*")

        gap_data = []
        for inc in sorted(df['income_quartile'].unique()):
            subset = df[df['income_quartile'] == inc]
            actual_rate = subset['funded_baseline'].mean()
            justified_rate = subset['risk_justified_prob'].mean() # We always show the gap relative to baseline
            gap = actual_rate - justified_rate
            gap_data.append({'Income Quartile': int(inc), 'Unexplained Gap': gap})

        gap_df = pd.DataFrame(gap_data)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        colors = ['#ff6b6b' if x < 0 else '#4ecdc4' for x in gap_df['Unexplained Gap']]
        sns.barplot(data=gap_df, x='Income Quartile', y='Unexplained Gap', ax=ax2, palette=colors)
        ax2.axhline(0, color='black', lw=1)
        ax2.set_ylabel("Gap vs Risk-Justified Rate")
        ax2.set_xlabel("Income Quartile (1=Lowest Income, 4=Highest Income)")
        st.pyplot(fig2)

        # Narrative interpretation
        lowest_q_gap = gap_df[gap_df['Income Quartile'] == 1]['Unexplained Gap'].iloc[0]
        if lowest_q_gap < -0.1:
            st.error(f"⚠️ **Unjustified Disparity Detected**: Quartile 1 is under-funded by {abs(lowest_q_gap):.1%} relative to their physical risk. The AI's financial logic (NOI/Value) is over-weighting wealthy coastal properties.")
        else:
            st.success("✅ Allocation appears reasonably aligned with physical risk across quartiles.")

    # --- Intervention / Trust Tax Section ---
    if apply_guardrail and tax is not None:
        st.markdown("---")
        st.subheader("3. Guardrail Impact: The Trust Tax")
        st.markdown("By enforcing parity, we must override the AI's strictly financial ranking. Here is the explicit cost of that decision:")
        tcol1, tcol2 = st.columns(2)
        tcol1.warning(f"### {tax['swaps_count']} Properties Swapped\n\nThe number of high-value/high-risk properties that were removed from the funding list to make room for lower-income properties.")
        tcol2.warning(f"### {tax['precision_drop']:.1%} Precision Drop\n\nThe percentage decrease in \"optimal\" financial efficiency compared to the baseline AI ranking.")

    st.markdown("---")

    # --- Map and Data Table ---
    st.subheader("4. Geographic View & Final List")
    st.markdown("Explore the physical locations of the funded properties.")

    # Map
    m = folium.Map(location=[26.5, -81.0], zoom_start=7)

    for idx, row in df_display.iterrows():
        is_funded = row[funded_col]
        color = 'green' if is_funded else 'gray'
        radius = 5 if is_funded else 2

        popup_text = f"""
        <b>{row['name']}</b><br>
        Value: ${row['property_value']:,.0f}<br>
        Income Quartile: {row['income_quartile']}<br>
        Flood Zone: {row['flood_zone']}<br>
        Risk Score: {row['physical_risk_score']:.2f}
        """

        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=popup_text
        ).add_to(m)

    st_folium(m, width=1200, height=400)

    st.markdown("### Approved Funding List")
    display_cols = ['name', 'county', 'income_quartile', 'flood_zone', 'physical_risk_score', 'financial_exposure_score', 'property_value']
    # Show only the funded properties, sorted by physical risk
    st.dataframe(
        df_display[df_display[funded_col]][display_cols].sort_values('physical_risk_score', ascending=False),
        use_container_width=True
    )
