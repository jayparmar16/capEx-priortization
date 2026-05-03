import pandas as pd
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
import math
import json
import time
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# We will implement the multi-agent system using LangGraph and Nvidia NIM.

class PropertyState(TypedDict):
    portfolio: pd.DataFrame
    scored_portfolio: pd.DataFrame
    logs: List[str]

import re

# Initialize the Nvidia NIM LLM
llm = ChatNVIDIA(
    model="z-ai/glm4.7",
    api_key="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",
    temperature=0.1, # Low temp for more consistent scoring
    top_p=1,
)

def batch_process_with_llm(prompt: str, data: list) -> list:
    """Helper to process a batch of data through the LLM."""
    full_prompt = f"{prompt}\n\nHere is the data in JSON format:\n{json.dumps(data)}\n\nProvide your response ONLY as a valid JSON array of objects matching the input order, with the newly computed scores."

    try:
        response = llm.invoke([{"role": "user", "content": full_prompt}])
        content = response.content.strip()

        # Robustly extract JSON array using regex
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            print(f"LLM Error: Could not find JSON array in response:\n{content[:100]}...")
            return []

    except Exception as e:
        print(f"LLM Error: {e}")
        # Fallback to deterministic if LLM fails formatting
        return []

def risk_assessor_agent(state: PropertyState) -> PropertyState:
    """Evaluates the physical risk based on wind and flood data using LLM."""
    df = state["portfolio"].copy()
    logs = state.get("logs", [])

    # Process in batches to avoid token limits
    batch_size = 20
    all_results = []

    system_prompt = """You are an expert structural engineer and risk assessor.
    Review the provided FEMA 'flood_zone' (VE is high risk, AE is moderate, X is low) and ASCE 7 'wind_risk_mph' for the following commercial properties.
    Calculate a 'physical_risk_score' between 0.0 and 1.5, weighting wind and flood equally. High coastal hazard zones (VE) and winds over 150mph should approach the maximum score.
    Return a JSON array of objects, each containing the 'id' and the computed 'physical_risk_score' as a float."""

    records = df[['id', 'wind_risk_mph', 'flood_zone']].to_dict('records')

    print(f"Risk Assessor evaluating {len(records)} properties via LLM...")
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        results = batch_process_with_llm(system_prompt, batch)
        if results:
            all_results.extend(results)
        else:
            # Deterministic Fallback if LLM parsing fails for this batch
            print(f"Fallback for batch {i}")
            for row in batch:
                wind_score = (row['wind_risk_mph'] - 100) / 100
                flood_score = 1.0 if row['flood_zone'] == 'VE' else (0.6 if row['flood_zone'] == 'AE' else 0.1)
                all_results.append({'id': row['id'], 'physical_risk_score': (wind_score * 0.5) + (flood_score * 0.5)})
        time.sleep(1) # Rate limiting

    # Merge results back
    results_df = pd.DataFrame(all_results)
    df = pd.merge(df, results_df, on='id', how='left')

    logs.append("Risk Assessor completed physical risk evaluation via Nvidia LLM.")
    return {"portfolio": state["portfolio"], "scored_portfolio": df, "logs": logs}

def financial_analyst_agent(state: PropertyState) -> PropertyState:
    """Evaluates the financial exposure using LLM."""
    df = state["scored_portfolio"].copy()
    logs = state.get("logs", [])

    batch_size = 20
    all_results = []

    system_prompt = """You are a commercial real estate financial analyst.
    Review the 'noi' (Net Operating Income) and 'insurance_premium' for the following properties.
    Output a normalized 'financial_exposure_score' between 0.0 and 1.0.
    Weight the NOI heavily (60%) as it represents cash flow at risk, but also consider the insurance burden (40%). Assume max NOI is ~2,000,000 and max premium is ~500,000 for normalization.
    Return a JSON array of objects, each containing the 'id' and the computed 'financial_exposure_score' as a float."""

    records = df[['id', 'noi', 'insurance_premium']].to_dict('records')

    print(f"Financial Analyst evaluating {len(records)} properties via LLM...")
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        results = batch_process_with_llm(system_prompt, batch)
        if results:
            all_results.extend(results)
        else:
            print(f"Fallback for batch {i}")
            for row in batch:
                noi_score = min(row['noi'] / 2_000_000, 1.0)
                ins_score = min(row['insurance_premium'] / 500_000, 1.0)
                all_results.append({'id': row['id'], 'financial_exposure_score': (noi_score * 0.6) + (ins_score * 0.4)})
        time.sleep(1)

    results_df = pd.DataFrame(all_results)
    df = pd.merge(df, results_df, on='id', how='left')

    logs.append("Financial Analyst completed exposure evaluation via Nvidia LLM.")
    return {"portfolio": state["portfolio"], "scored_portfolio": df, "logs": logs}

def prioritization_scorer_agent(state: PropertyState) -> PropertyState:
    """Combines risk and financial scores into a final 'fund-first' ranking using LLM."""
    df = state["scored_portfolio"].copy()
    logs = state.get("logs", [])

    batch_size = 20
    all_results = []

    system_prompt = """You are the lead portfolio asset manager.
    Review the 'physical_risk_score' and 'financial_exposure_score'.
    Calculate a 'final_priority_score' by weighting the financial exposure at 60% and the physical risk at 40%.
    Return a JSON array of objects, each containing the 'id' and the computed 'final_priority_score' as a float."""

    records = df[['id', 'physical_risk_score', 'financial_exposure_score']].to_dict('records')

    print(f"Prioritization Scorer evaluating {len(records)} properties via LLM...")
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        results = batch_process_with_llm(system_prompt, batch)
        if results:
            all_results.extend(results)
        else:
            print(f"Fallback for batch {i}")
            for row in batch:
                score = (row['financial_exposure_score'] * 0.6) + (row['physical_risk_score'] * 0.4)
                all_results.append({'id': row['id'], 'final_priority_score': score})
        time.sleep(1)

    results_df = pd.DataFrame(all_results)
    df = pd.merge(df, results_df, on='id', how='left')

    # LLMs handle the scoring, but pandas is better at strict ranking across the whole set
    df['baseline_rank'] = df['final_priority_score'].rank(ascending=False, method='min')
    df = df.sort_values(by='baseline_rank').reset_index(drop=True)

    # Top 30% of portfolio
    top_k = math.ceil(len(df) * 0.3)
    df['funded_baseline'] = False
    df.loc[:top_k-1, 'funded_baseline'] = True

    logs.append(f"Prioritization Scorer ranked properties and selected Top {top_k} for funding.")

    return {"portfolio": state["portfolio"], "scored_portfolio": df, "logs": logs}

def run_agentic_scorer(csv_path: str, output_path: str):
    df = pd.read_csv(csv_path)

    # Build LangGraph workflow
    workflow = StateGraph(PropertyState)

    workflow.add_node("risk_assessor", risk_assessor_agent)
    workflow.add_node("financial_analyst", financial_analyst_agent)
    workflow.add_node("prioritization_scorer", prioritization_scorer_agent)

    workflow.set_entry_point("risk_assessor")
    workflow.add_edge("risk_assessor", "financial_analyst")
    workflow.add_edge("financial_analyst", "prioritization_scorer")
    workflow.add_edge("prioritization_scorer", END)

    app = workflow.compile()

    initial_state = {"portfolio": df, "scored_portfolio": pd.DataFrame(), "logs": []}
    final_state = app.invoke(initial_state)

    scored_df = final_state["scored_portfolio"]
    scored_df.to_csv(output_path, index=False)
    print("\n".join(final_state["logs"]))
    print(f"Scored portfolio saved to {output_path}")

if __name__ == "__main__":
    run_agentic_scorer("data/portfolio_augmented.csv", "data/portfolio_scored.csv")
