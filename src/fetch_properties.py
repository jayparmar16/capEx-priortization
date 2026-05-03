import requests
import json
import time
import pandas as pd
import numpy as np
import random
from geopy.geocoders import Nominatim

# We'll use Overpass API to get random commercial properties in Miami-Dade, Broward, and Polk Counties.
# Polk is a good inland county for contrast (lower income on avg, lower coastal flood risk but some inland flood, lower wind risk compared to Miami-Dade/Broward).
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

COUNTIES = {
    "Miami-Dade": "Miami-Dade County, Florida",
    "Broward": "Broward County, Florida",
    "Polk": "Polk County, Florida"
}

def get_bounding_box(location_name):
    geolocator = Nominatim(user_agent="florida_cre_audit_pipeline")
    location = geolocator.geocode(location_name)
    if location and location.raw.get('boundingbox'):
        # bbox is [lat_min, lat_max, lon_min, lon_max]
        bbox = location.raw['boundingbox']
        return f"{bbox[0]},{bbox[2]},{bbox[1]},{bbox[3]}"
    return None

def fetch_commercial_properties(county_name, limit=50):
    bbox = get_bounding_box(COUNTIES[county_name])
    if not bbox:
        print(f"Could not find bbox for {county_name}")
        return []

    # Overpass query: looking for ways/relations that are tagged as commercial buildings or offices
    overpass_query = f"""
    [out:json];
    (
      way["building"="commercial"]({bbox});
      way["building"="office"]({bbox});
      way["building"="retail"]({bbox});
    );
    out center {limit * 3};
    """

    print(f"Fetching properties for {county_name}...")
    # Overpass sometimes rejects the default requests user-agent
    headers = {"User-Agent": "FloridaCREAudit/1.0 (test@example.com)"}
    response = requests.post(OVERPASS_URL, data={"data": overpass_query}, headers=headers)

    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code} - {response.text}")
        return []

    data = response.json()
    elements = data.get('elements', [])

    properties = []
    for el in elements:
        if el['type'] == 'way' and 'center' in el:
            tags = el.get('tags', {})
            properties.append({
                'id': f"osm_{el['id']}",
                'lat': el['center']['lat'],
                'lon': el['center']['lon'],
                'county': county_name,
                'building_type': tags.get('building', 'commercial'),
                'name': tags.get('name', f"Commercial Property {el['id']}")
            })

    # Randomly sample 'limit' number of properties to avoid clustering too much
    if len(properties) > limit:
        properties = random.sample(properties, limit)

    return properties

# Generate Mock Portfolio
portfolio = []
for county in COUNTIES.keys():
    props = fetch_commercial_properties(county, limit=50)
    portfolio.extend(props)
    time.sleep(2) # be nice to Overpass API

df = pd.DataFrame(portfolio)
print(f"Fetched {len(df)} properties.")

df.to_csv("data/raw_properties.csv", index=False)
