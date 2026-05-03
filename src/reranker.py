import pandas as pd
import numpy as np
import math

def parity_constrained_rerank(df: pd.DataFrame, target_k: int, tolerance: float = 0.05):
    """
    Adjusts the fund-first list to keep selection rates across income groups within
    a configurable tolerance of the overall average selection rate, while staying
    as close as possible to the original risk-weighted ranking.
    """
    df = df.copy()

    overall_selection_rate = target_k / len(df)
    target_min = overall_selection_rate - tolerance
    target_max = overall_selection_rate + tolerance

    # We will maintain a list of currently selected indices
    selected_indices = list(df.head(target_k).index)
    unselected_indices = list(df.tail(len(df) - target_k).index)

    # Track metrics for the trust tax
    high_risk_bumped = 0
    swaps_made = []

    def check_parity(selected_list):
        temp_df = df.copy()
        temp_df['temp_funded'] = False
        temp_df.loc[selected_list, 'temp_funded'] = True

        rates = temp_df.groupby('income_quartile')['temp_funded'].mean().to_dict()
        violators_under = [inc for inc, rate in rates.items() if rate < target_min]
        violators_over = [inc for inc, rate in rates.items() if rate > target_max]

        return violators_under, violators_over, rates

    max_iterations = 100
    for _ in range(max_iterations):
        violators_under, violators_over, current_rates = check_parity(selected_indices)

        if not violators_under and not violators_over:
            break # Parity achieved

        if violators_under and violators_over:
            # Swap an over-represented low-rank selected item for an under-represented high-rank unselected item

            # Find the lowest ranked item currently selected that belongs to an over-represented group
            over_group_items = df.loc[selected_indices]
            over_group_items = over_group_items[over_group_items['income_quartile'].isin(violators_over)]

            if over_group_items.empty:
                break

            item_to_remove = over_group_items.sort_values('final_priority_score').index[0] # The one with the lowest score

            # Find the highest ranked item currently UNselected that belongs to an under-represented group
            under_group_items = df.loc[unselected_indices]
            under_group_items = under_group_items[under_group_items['income_quartile'].isin(violators_under)]

            if under_group_items.empty:
                break

            item_to_add = under_group_items.sort_values('final_priority_score', ascending=False).index[0] # Highest score

            # Swap
            selected_indices.remove(item_to_remove)
            selected_indices.append(item_to_add)
            unselected_indices.remove(item_to_add)
            unselected_indices.append(item_to_remove)

            swaps_made.append((item_to_remove, item_to_add))
            high_risk_bumped += 1

        else:
            # If we only have under or over, we might need to adjust target_k slightly or relax constraints
            # For this MVP, we just break if we can't do a clean swap
            break

    df['funded_reranked'] = False
    df.loc[selected_indices, 'funded_reranked'] = True

    # Calculate Trust Tax
    baseline_selected = set(df[df['funded_baseline']].index)
    rerank_selected = set(selected_indices)

    overlap = len(baseline_selected.intersection(rerank_selected))
    precision_drop = (target_k - overlap) / target_k

    return df, {
        "swaps_count": high_risk_bumped,
        "precision_drop": precision_drop,
        "swaps_details": swaps_made
    }

if __name__ == "__main__":
    df = pd.read_csv("data/portfolio_audited.csv")
    target_k = int(df['funded_baseline'].sum())

    df_rerank, tax = parity_constrained_rerank(df, target_k, tolerance=0.10)

    print(f"Parity Rerank Completed.")
    print(f"Trust Tax - High-Risk Properties Bumped (Swaps): {tax['swaps_count']}")
    print(f"Trust Tax - Precision Drop relative to pure risk/finance rank: {tax['precision_drop']:.2%}")

    rates_before = df.groupby('income_quartile')['funded_baseline'].mean()
    rates_after = df_rerank.groupby('income_quartile')['funded_reranked'].mean()

    print("\nRates Before:")
    print(rates_before)
    print("\nRates After:")
    print(rates_after)

    df_rerank.to_csv("data/portfolio_reranked.csv", index=False)
