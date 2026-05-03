import pandas as pd
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
import math

# We will implement a mock multi-agent system using LangGraph.
# In a real scenario, these agents would call external LLMs or rules engines.
# Here, we use deterministic risk and financial scoring functions to represent the AI's output.

class PropertyState(TypedDict):
    portfolio: pd.DataFrame
    scored_portfolio: pd.DataFrame
    logs: List[str]

def risk_assessor_agent(state: PropertyState) -> PropertyState:
    """Evaluates the physical risk based on wind and flood data."""
    df = state["portfolio"].copy()
    logs = state.get("logs", [])

    # Simple risk heuristic: high wind + high flood = high risk score
    def calculate_risk(row):
        wind_score = (row['wind_risk_mph'] - 100) / 100  # Normalize roughly
        flood_score_map = {'VE': 1.0, 'AE': 0.6, 'X': 0.1}
        flood_score = flood_score_map.get(row['flood_zone'], 0.1)

        return (wind_score * 0.5) + (flood_score * 0.5)

    df['physical_risk_score'] = df.apply(calculate_risk, axis=1)
    logs.append("Risk Assessor completed physical risk evaluation.")

    return {"portfolio": state["portfolio"], "scored_portfolio": df, "logs": logs}

def financial_analyst_agent(state: PropertyState) -> PropertyState:
    """Evaluates the financial exposure (NOI at risk, property value)."""
    df = state["scored_portfolio"].copy()
    logs = state.get("logs", [])

    def calculate_financial_exposure(row):
        # Normalize NOI roughly (assume max is around $2M)
        noi_score = min(row['noi'] / 2_000_000, 1.0)
        # Normalize insurance premium (assume max is around $500k)
        ins_score = min(row['insurance_premium'] / 500_000, 1.0)

        return (noi_score * 0.6) + (ins_score * 0.4)

    df['financial_exposure_score'] = df.apply(calculate_financial_exposure, axis=1)
    logs.append("Financial Analyst completed exposure evaluation.")

    return {"portfolio": state["portfolio"], "scored_portfolio": df, "logs": logs}

def prioritization_scorer_agent(state: PropertyState) -> PropertyState:
    """Combines risk and financial scores into a final 'fund-first' ranking."""
    df = state["scored_portfolio"].copy()
    logs = state.get("logs", [])

    # Final Score: 60% Financial Exposure, 40% Physical Risk (Demographic-blind)
    df['final_priority_score'] = (df['financial_exposure_score'] * 0.6) + (df['physical_risk_score'] * 0.4)

    # Rank descending
    df['baseline_rank'] = df['final_priority_score'].rank(ascending=False, method='min')
    df = df.sort_values(by='baseline_rank').reset_index(drop=True)

    # We define the "fund-first" list as the Top K (e.g., top 30% of portfolio)
    top_k = math.ceil(len(df) * 0.3)
    df['funded_baseline'] = False
    df.loc[:top_k-1, 'funded_baseline'] = True

    logs.append(f"Prioritization Scorer ranked properties and selected Top {top_k} for funding.")

    return {"portfolio": state["portfolio"], "scored_portfolio": df, "logs": logs}

def run_agentic_scorer(csv_path: str, output_path: str):
    df = pd.read_csv(csv_path)

    # Build LangGraph workflow
    workflow = StateGraph(PropertyState)

    # Add nodes
    workflow.add_node("risk_assessor", risk_assessor_agent)
    workflow.add_node("financial_analyst", financial_analyst_agent)
    workflow.add_node("prioritization_scorer", prioritization_scorer_agent)

    # Add edges
    workflow.set_entry_point("risk_assessor")
    workflow.add_edge("risk_assessor", "financial_analyst")
    workflow.add_edge("financial_analyst", "prioritization_scorer")
    workflow.add_edge("prioritization_scorer", END)

    # Compile
    app = workflow.compile()

    # Execute
    initial_state = {"portfolio": df, "scored_portfolio": pd.DataFrame(), "logs": []}
    final_state = app.invoke(initial_state)

    scored_df = final_state["scored_portfolio"]
    scored_df.to_csv(output_path, index=False)
    print("\n".join(final_state["logs"]))
    print(f"Scored portfolio saved to {output_path}")

if __name__ == "__main__":
    run_agentic_scorer("data/portfolio_augmented.csv", "data/portfolio_scored.csv")
