import pandas as pd
import numpy as np
import requests
import time
import urllib.parse

# A script to augment the base properties with REAL Census demographics, REAL FEMA flood zones,
# and synthetic financial/wind proxies.

def get_fips_from_latlon(lat, lon):
    """Uses the FCC Area API to convert lat/lon to a Census FIPS block code."""
    url = f"https://geo.fcc.gov/api/census/block/find?latitude={lat}&longitude={lon}&format=json"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            # The FIPS code is 15 digits: State (2), County (3), Tract (6), Block Group (1), Block (3)
            # We need State (2) + County (3) + Tract (6) for the ACS API.
            fips = data.get('Block', {}).get('FIPS', '')
            if len(fips) >= 11:
                state = fips[0:2]
                county = fips[2:5]
                tract = fips[5:11]
                return state, county, tract
    except Exception as e:
        print(f"FCC API error for {lat}, {lon}: {e}")
    return None, None, None

def get_census_demographics(state, county, tract):
    """
    Uses the US Census Bureau API (2022 ACS 5-Year Estimates) to get:
    - Median Household Income (B19013_001E)
    - Total Population (B02001_001E)
    - White Alone Population (B02001_002E) -> Used to calculate Minority Pct
    """
    # Variables: B19013_001E (Median Income), B02001_001E (Total Pop), B02001_002E (White Alone Pop)
    url = f"https://api.census.gov/data/2022/acs/acs5?get=B19013_001E,B02001_001E,B02001_002E&for=tract:{tract}&in=state:{state}%20county:{county}"

    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if len(data) > 1: # Row 0 is headers, Row 1 is data
                income = int(data[1][0]) if data[1][0] and int(data[1][0]) > 0 else None
                total_pop = int(data[1][1]) if data[1][1] else 0
                white_pop = int(data[1][2]) if data[1][2] else 0

                minority_pct = ((total_pop - white_pop) / total_pop) if total_pop > 0 else 0.5
                return income, minority_pct
    except Exception as e:
        pass

    return None, None

def get_fema_flood_zone(lat, lon):
    """
    Queries the FEMA National Flood Hazard Layer (NFHL) ArcGIS REST API.
    """
    # We create a small bounding box around the point to query the spatial feature
    offset = 0.0005
    bbox = f"{lon-offset},{lat-offset},{lon+offset},{lat+offset}"

    # Layer 28 in FEMA NFHL is 'Flood Hazard Zones'
    url = f"https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer/28/query"
    params = {
        'geometry': bbox,
        'geometryType': 'esriGeometryEnvelope',
        'spatialRel': 'esriSpatialRelIntersects',
        'outFields': 'FLD_ZONE',
        'returnGeometry': 'false',
        'f': 'json'
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            features = data.get('features', [])
            if features:
                # Return the zone of the first intersecting feature
                zone = features[0].get('attributes', {}).get('FLD_ZONE', 'X')

                # Simplify FEMA zones to our 3 main categories for the scorer
                if 'V' in zone:
                    return 'VE'
                elif 'A' in zone:
                    return 'AE'
                else:
                    return 'X'
    except Exception as e:
        print(f"FEMA API error: {e}")

    return 'X' # Default to minimal risk if API fails or no data

def augment_properties():
    df = pd.read_csv("data/raw_properties.csv")
    np.random.seed(42)

    print("Fetching real Demographics and Flood Zones via API. This will take a few minutes...")

    incomes = []
    minorities = []
    flood_zones = []

    for idx, row in df.iterrows():
        lat, lon = row['lat'], row['lon']

        # 1. Fetch Demographics
        state, county, tract = get_fips_from_latlon(lat, lon)
        income = None
        minority = None
        if state and county and tract:
            income, minority = get_census_demographics(state, county, tract)

        incomes.append(income)
        minorities.append(minority)

        # 2. Fetch Flood Zone
        zone = get_fema_flood_zone(lat, lon)
        flood_zones.append(zone)

        # Print progress
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df)} properties...")

        time.sleep(0.5) # Be nice to public APIs

    df['raw_income'] = incomes
    df['minority_pct'] = minorities
    df['flood_zone'] = flood_zones

    # Handle missing API data with realistic fallbacks
    median_income = df['raw_income'].median()
    df['raw_income'] = df['raw_income'].fillna(median_income)
    df['minority_pct'] = df['minority_pct'].fillna(df['minority_pct'].mean())

    # Convert raw income to quartiles for the dashboard
    df['income_quartile'] = pd.qcut(df['raw_income'], q=4, labels=[1, 2, 3, 4]).astype(float)

    # 3. Add Risk Proxies (ASCE Wind)
    # Wind risk is higher on the coast (Miami-Dade/Broward)
    def assign_wind(row):
        if row['county'] in ['Miami-Dade', 'Broward']:
            return np.random.uniform(150, 185)
        else:
            return np.random.uniform(110, 140)

    df['wind_risk_mph'] = df.apply(assign_wind, axis=1)

    # 4. Add Financials (Yardi/MRI proxy)
    # Real income now drives the synthetic property value.
    def assign_financials(row):
        base_value = 5_000_000

        # Use real income to drive value proxy
        inc_mult = row['raw_income'] / 60000.0 # Normalized against 60k average

        county_mult = 1.3 if row['county'] in ['Miami-Dade', 'Broward'] else 0.8
        noise = np.random.uniform(0.9, 1.1)

        property_value = base_value * county_mult * inc_mult * noise

        cap_rate = np.random.uniform(0.05, 0.08)
        noi = property_value * cap_rate

        # Insurance burden is real flood zone dependent
        ins_rate = 0.03 if row['flood_zone'] == 'VE' else (0.015 if row['flood_zone'] == 'AE' else 0.008)
        ins_premium = property_value * ins_rate

        return pd.Series([property_value, noi, ins_premium])

    df[['property_value', 'noi', 'insurance_premium']] = df.apply(assign_financials, axis=1)

    df.to_csv("data/portfolio_augmented.csv", index=False)
    print(f"Finished augmenting {len(df)} properties with REAL API data and financial proxies.")

if __name__ == "__main__":
    augment_properties()
