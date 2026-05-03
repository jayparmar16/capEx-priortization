import pandas as pd
import numpy as np

# A script to augment the base properties with mock demographics, risk scores, and financials.
# This ensures our pipeline has the realistic, multi-layered data needed.

def augment_properties():
    df = pd.read_csv("data/raw_properties.csv")

    np.random.seed(42)

    # 1. Add Census Demographics (Mocked realistically based on geography)
    # Coastal counties (Miami-Dade, Broward) have mixed income but many high-value/high-income tracts near the water.
    # Polk (inland) is generally lower income, higher proportion of working-class neighborhoods.
    def assign_demographics(row):
        if row['county'] in ['Miami-Dade', 'Broward']:
            # Assume 40% high income, 30% upper-mid, 20% lower-mid, 10% low
            inc_quartile = np.random.choice([4, 3, 2, 1], p=[0.4, 0.3, 0.2, 0.1])
            # Racial composition: simplified minority percentage
            minority_pct = np.random.uniform(0.1, 0.9) if inc_quartile <= 2 else np.random.uniform(0.05, 0.5)
        else:
            # Polk
            inc_quartile = np.random.choice([4, 3, 2, 1], p=[0.1, 0.2, 0.4, 0.3])
            minority_pct = np.random.uniform(0.2, 0.8) if inc_quartile <= 2 else np.random.uniform(0.1, 0.4)

        return pd.Series([inc_quartile, minority_pct])

    df[['income_quartile', 'minority_pct']] = df.apply(assign_demographics, axis=1)

    # 2. Add Risk Proxies (FEMA Flood & ASCE Wind)
    # High-income coastal tracts tend to be VE (High Hazard Coastal) or AE.
    # Inland tracts are X (Minimal) or AE (riverine flood).
    # Wind risk is higher on the coast (e.g., 170+ mph vs 130 mph inland).
    def assign_risk(row):
        if row['county'] in ['Miami-Dade', 'Broward']:
            wind_risk_mph = np.random.uniform(160, 185)
            # Higher property value / income on the coast correlates with VE zones
            if row['income_quartile'] >= 3:
                flood_zone = np.random.choice(['VE', 'AE', 'X'], p=[0.5, 0.3, 0.2])
            else:
                flood_zone = np.random.choice(['VE', 'AE', 'X'], p=[0.1, 0.4, 0.5])
        else:
            wind_risk_mph = np.random.uniform(120, 140)
            flood_zone = np.random.choice(['AE', 'X'], p=[0.3, 0.7])

        return pd.Series([flood_zone, wind_risk_mph])

    df[['flood_zone', 'wind_risk_mph']] = df.apply(assign_risk, axis=1)

    # 3. Add Financials (Yardi/MRI mock)
    # NOI (Net Operating Income) and Property Value correlate strongly with location and income quartile.
    def assign_financials(row):
        base_value = 5_000_000 # $5M base commercial

        # Multipliers based on income area and county
        county_mult = 1.5 if row['county'] in ['Miami-Dade', 'Broward'] else 0.8
        inc_mult = {4: 2.0, 3: 1.5, 2: 1.0, 1: 0.7}[row['income_quartile']]

        # Add random noise
        noise = np.random.uniform(0.8, 1.2)

        property_value = base_value * county_mult * inc_mult * noise

        # Cap rate proxy: 5-8%
        cap_rate = np.random.uniform(0.05, 0.08)
        noi = property_value * cap_rate

        # Insurance burden is higher in VE zones
        ins_rate = 0.03 if row['flood_zone'] == 'VE' else (0.015 if row['flood_zone'] == 'AE' else 0.008)
        ins_premium = property_value * ins_rate

        return pd.Series([property_value, noi, ins_premium])

    df[['property_value', 'noi', 'insurance_premium']] = df.apply(assign_financials, axis=1)

    df.to_csv("data/portfolio_augmented.csv", index=False)
    print(f"Augmented {len(df)} properties with demographics, risk, and financials.")

if __name__ == "__main__":
    augment_properties()
