# Execution Runbook: In-Depth Data Flow

This document details the exact flow of data through the AI governance pipeline, explaining what each script does, where it gets its data, and where it stores its output.

All commands assume you are running them from the root of the repository with your Python virtual environment activated.

## Directory Structure Overview

* `src/`: Contains all the Python scripts for the pipeline and the dashboard.
* `data/`: The directory where all intermediate and final CSV files are stored. **Note: This directory is ignored by Git, so these files will only exist locally after you run the pipeline.**
* `notebooks/`: Contains the Jupyter notebook for the final narrative analysis.

---

## Step 1: Data Acquisition
**Command:** `python src/fetch_properties.py`

* **What it does:** Uses the public Overpass API to search OpenStreetMap for commercial buildings in Miami-Dade, Broward, and Polk counties. It extracts their latitude, longitude, and basic metadata.
* **Input:** None (queries an external public API).
* **Output:** Saves the raw property coordinates to `data/raw_properties.csv`.

## Step 2: Data Augmentation (Mock Demographics & Financials)
**Command:** `python src/augment_data.py`

* **What it does:** Reads the raw properties and synthetically generates realistic data layers based on the property's geography (e.g., coastal tracts get higher property values, higher wind risk, and different income distributions compared to inland tracts).
* **Input:** Reads `data/raw_properties.csv`.
* **Output:** Saves the enriched dataset to `data/portfolio_augmented.csv`. This file now contains columns like `income_quartile`, `minority_pct`, `flood_zone`, `wind_risk_mph`, `property_value`, and `noi`.

## Step 3: Multi-Agent Prioritization (The Baseline Scorer)
**Command:** `python src/agentic_scorer.py`

* **What it does:** Executes the LangGraph multi-agent workflow. The *Risk Assessor Agent* calculates physical risk based on wind and flood data. The *Financial Analyst Agent* calculates exposure based on property value and NOI. The *Prioritization Scorer Agent* combines these scores and selects the top 30% of properties to be "funded."
* **Input:** Reads `data/portfolio_augmented.csv`.
* **Output:** Saves the scored and ranked dataset to `data/portfolio_scored.csv`. This file includes new columns like `physical_risk_score`, `financial_exposure_score`, and the boolean flag `funded_baseline`.

## Step 4: The Audit Pipeline
**Command:** `python src/audit_pipeline.py`

* **What it does:** Analyzes the baseline scoring results. It calculates raw selection rates by demographic group. Crucially, it runs a Logistic Regression to predict the probability of funding based *only* on physical risk, calculating the "unexplained gap" between actual funding and risk-justified funding.
* **Input:** Reads `data/portfolio_scored.csv`.
* **Output:** Prints the metrics to the terminal and saves the dataset (now including the `risk_justified_prob` column) to `data/portfolio_audited.csv`.

## Step 5: The Parity-Constrained Re-ranker
**Command:** `python src/reranker.py`

* **What it does:** Reads the audited data and attempts to "fix" the demographic disparities. It swaps highly-ranked properties from over-represented groups with lower-ranked properties from under-represented groups until demographic selection rates fall within a 10% tolerance. It calculates the "Trust Tax" (how many swaps were required and the drop in precision).
* **Input:** Reads `data/portfolio_audited.csv`.
* **Output:** Prints the Trust Tax metrics to the terminal and saves the final re-ranked dataset to `data/portfolio_reranked.csv` (which includes the new boolean flag `funded_reranked`).

---

## Final Consumption

Once the pipeline has generated the data, there are two ways to consume the results:

### 1. The Fairness Dashboard
**Command:** `streamlit run src/dashboard.py`
* **Input:** The dashboard dynamically reads `data/portfolio_audited.csv`. When the user toggles the "Enable Parity Re-ranker" checkbox, it dynamically calls the `parity_constrained_rerank` function from `src/reranker.py` to calculate the swaps in real-time.

### 2. The Jupyter Notebook Analysis
**Command:** `jupyter notebook notebooks/analysis.ipynb`
* **Input:** The notebook reads `../data/portfolio_audited.csv` and also imports the `parity_constrained_rerank` function from `../src/reranker.py` to generate its visualizations and narrative.