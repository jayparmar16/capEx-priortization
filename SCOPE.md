# Scope & Limitations

This project is a functional proof-of-concept for an AI governance layer. It is designed to demonstrate the *methodology* of algorithmic auditing and mitigation in commercial real estate.

## What is In-Scope
* **Realistic Geography**: Fetching actual commercial building coordinates via OpenStreetMap's Overpass API from three contrasting Florida counties:
  * Miami-Dade (Coastal, High-Value, High-Risk)
  * Broward (Coastal, High-Value, High-Risk)
  * Polk (Inland, Mixed-Value, Lower-Risk)
* **Synthetic Demographics & Risk**: Simulating Census tract income quartiles, minority percentages, FEMA flood zones, and wind risk using distributions that mirror reality (e.g., coastal areas having higher property values and higher wind risk).
* **Synthetic Financials**: Generating Net Operating Income (NOI) and insurance premiums based on property value, location, and risk zone.
* **Deterministic Agentic Scoring**: Simulating LLM/rules-engine agents with deterministic Python functions for predictable auditing.
* **Interactive Dashboard**: A local Streamlit application to visualize the output and control the re-ranker.

## What is Out-of-Scope (Limitations)
* **Live API Integrations for Private Data**: Real Yardi/MRI financial data is private. Real property-level ASCE 7 wind codes require expensive commercial APIs (e.g., CoreLogic, RMS). We use proxy distributions instead.
* **Live Census API Calls**: To ensure the pipeline runs smoothly without requiring API keys, demographic data is synthetically assigned based on the county's general profile rather than mapping the precise lat/lon to a live US Census API endpoint.
* **Complex LLM Reasoning**: The LangGraph agents currently use hardcoded mathematical heuristics. In a full production system, these nodes would likely wrap calls to GPT-4 or Claude to parse unstructured PDF risk reports or insurance filings.
* **Scalability**: The Overpass API limits the number of properties we can reliably fetch without timing out. The pipeline is currently scoped to a portfolio of ~150 properties to simulate a mid-market asset manager.
* **Authentication**: The Streamlit dashboard is designed for local execution and does not include user authentication or role-based access control.