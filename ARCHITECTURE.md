# Architecture & LangGraph Implementation

## The Multi-Agent Scorer

The baseline prioritization algorithm is implemented as a stateful graph using **LangGraph**. This simulates how enterprise AI systems aggregate different signals.

### Graph State
The state (`PropertyState`) passed between agents contains:
* `portfolio`: The raw property dataframe.
* `scored_portfolio`: The progressively updated dataframe with intermediate scores.
* `logs`: Execution logs tracking agent actions.

### Agent Nodes
1. **Risk Assessor Agent**:
   * **Input**: Physical location data, FEMA flood zones (VE, AE, X), and ASCE 7 wind risk (mph).
   * **Output**: Computes a normalized `physical_risk_score`.
2. **Financial Analyst Agent**:
   * **Input**: Property Value, Net Operating Income (NOI), and Insurance Premium.
   * **Output**: Computes a normalized `financial_exposure_score`.
3. **Prioritization Scorer Agent**:
   * **Input**: Outputs from the previous two agents.
   * **Output**: Combines the scores (e.g., 60% Financial, 40% Physical Risk), ranks the portfolio descending, and selects the top 30% as the `funded_baseline` list.

## The Audit Logic (Risk-Adjusted Gap)

The core architectural innovation in the audit pipeline is isolating *unjustified* disparities.

1. **Logistic Regression Baseline**: We train a model to predict `funded_baseline` using *only* the `physical_risk_score`. This yields the probability that a property would be funded if the decision were based strictly on hurricane/flood risk.
2. **Gap Calculation**: We subtract this risk-justified probability from the *actual* selection rate for each demographic group.
3. **Interpretation**: A positive gap means the group is over-funded relative to their physical risk (usually driven by high property values). A negative gap means they are systematically under-funded.

## Parity Re-ranker

The re-ranker operates via a greedy swap algorithm:
1. It calculates the overall portfolio selection rate.
2. It identifies demographic groups falling below a configurable tolerance threshold.
3. It swaps the lowest-ranked selected properties from over-represented groups with the highest-ranked unselected properties from under-represented groups until parity is achieved.
4. It logs the number of swaps and the drop in top-K precision as the explicit "Trust Tax."