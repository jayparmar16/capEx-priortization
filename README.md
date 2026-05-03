# Florida CRE AI Governance Layer

This repository implements an AI governance and fairness layer for commercial real estate (CRE) portfolio prioritization in Florida.

Asset managers use multi-agent systems to decide which properties receive capital for hurricane and flood hardening. However, because property values, income levels, and risk zones are spatially correlated, purely financial/risk-based algorithms often create unintended demographic disparities.

This project provides tools to audit those algorithms, surface the hidden "Trust Tax" of mitigation, and empower asset managers to make informed allocation decisions.

## Components

1. **Mock Portfolio Generator**: Fetches real commercial parcel coordinates in Florida (Miami-Dade, Broward, Polk) via the Overpass API and augments them with realistic demographics, physical risk proxies (FEMA/ASCE), and mock financials (Yardi/MRI proxies).
2. **Agentic Scorer (LangGraph)**: A multi-agent workflow that ranks properties based purely on financial exposure and physical risk.
3. **Audit Pipeline**: Calculates raw demographic selection rates and uses logistic regression to compute the **Risk-Adjusted Gap**—isolating disparities caused by the financial logic versus genuine physical risk.
4. **Parity Re-ranker**: Adjusts the baseline ranking to meet demographic parity constraints, explicitly calculating the cost to precision (the "Trust Tax").
5. **Fairness Dashboard**: An interactive Streamlit application for asset managers.

## How to Run

### Prerequisites
Requires Python 3.9+.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt # (or install dependencies manually as defined in the code)
```

### 1. Run the Pipeline
Execute the pipeline sequentially to generate the data, score it, audit it, and calculate the re-ranking:

```bash
python src/fetch_properties.py
python src/augment_data.py
python src/agentic_scorer.py
python src/audit_pipeline.py
python src/reranker.py
```
*Note: The generated CSV files will be saved in the `data/` directory (which is git-ignored).*

### 2. View the Dashboard
Start the interactive Streamlit dashboard:

```bash
streamlit run src/dashboard.py
```

### 3. Read the Analysis
Open the Jupyter Notebook for a detailed narrative and visual breakdown of the findings:

```bash
jupyter notebook notebooks/analysis.ipynb
```
