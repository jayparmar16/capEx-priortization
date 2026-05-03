# Scope & Limitations

This project is a functional proof-of-concept for an AI governance layer. It is designed to demonstrate the *methodology* of algorithmic auditing and mitigation in commercial real estate.

## What is In-Scope
* **Realistic Geography**: Fetching actual commercial building coordinates via OpenStreetMap's Overpass API from three contrasting Florida counties:
  * Miami-Dade (Coastal, High-Value, High-Risk)
  * Broward (Coastal, High-Value, High-Risk)
  * Polk (Inland, Mixed-Value, Lower-Risk)
* **Real Demographics**: Querying the US Census Bureau API (ACS 5-Year Estimates) and FCC Area API to map exactly which Census Tract each property is in, pulling real Median Household Income and exact Minority Population percentages.
* **Real Flood Risk**: Querying the FEMA National Flood Hazard Layer (NFHL) ArcGIS REST API to identify the exact FEMA Flood Zone (e.g., VE, AE, X) for the physical coordinates of the property.
* **Synthetic Financials & Wind**: Generating Net Operating Income (NOI), insurance premiums, and ASCE-7 wind exposure. *Note: The synthetic financial values are now mathematically driven by the REAL Census income data to ensure realistic distribution.*
* **Deterministic Agentic Scoring**: Simulating LLM/rules-engine agents with deterministic Python functions for predictable auditing.
* **Interactive Dashboard**: A local Streamlit application to visualize the output and control the re-ranker.

## What is Out-of-Scope (Limitations)
* **Live API Integrations for Private Data**: Real Yardi/MRI financial data is private. Real property-level ASCE 7 wind codes require expensive commercial APIs (e.g., CoreLogic, RMS). We use proxy distributions instead.
* **Complex LLM Reasoning**: The LangGraph agents currently use hardcoded mathematical heuristics. In a full production system, these nodes would likely wrap calls to GPT-4 or Claude to parse unstructured PDF risk reports or insurance filings.
* **Scalability**: The Overpass API limits the number of properties we can reliably fetch without timing out. The pipeline is currently scoped to a portfolio of ~150 properties to simulate a mid-market asset manager.
* **Authentication**: The Streamlit dashboard is designed for local execution and does not include user authentication or role-based access control.