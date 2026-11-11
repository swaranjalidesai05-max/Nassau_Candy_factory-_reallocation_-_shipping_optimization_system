"""
Shared geographic coordinates, distance calculations, and baseline factory network
definitions for the Nassau Candy Distributor Optimization System.

Note: The raw transactional CSV does not contain factory coordinates or freight tariffs.
Factory locations, baseline product-to-factory assignments, capacities, and freight rates
below are explicitly parameterized modeling assumptions used for network simulation
and optimization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Approximate geographic centroids (latitude, longitude) for all 59 US states/DC
# and Canadian provinces present in the Nassau Candy Distributor dataset.
STATE_CENTROIDS: dict[str, tuple[float, float]] = {
    "Alabama": (32.806671, -86.791130),
    "Alberta": (53.933271, -116.576504),
    "Arizona": (33.729759, -111.431221),
    "Arkansas": (34.969704, -92.373123),
    "British Columbia": (53.726668, -127.647621),
    "California": (36.116203, -119.681564),
    "Colorado": (39.059811, -105.311104),
    "Connecticut": (41.597782, -72.755371),
    "Delaware": (39.318523, -75.507141),
    "District of Columbia": (38.897438, -77.026817),
    "Florida": (27.766279, -81.686783),
    "Georgia": (33.040619, -83.643074),
    "Idaho": (44.240459, -114.478828),
    "Illinois": (40.349457, -88.986137),
    "Indiana": (39.849426, -86.258278),
    "Iowa": (42.011539, -93.210526),
    "Kansas": (38.526600, -96.726486),
    "Kentucky": (37.668140, -84.670067),
    "Louisiana": (31.169546, -91.867805),
    "Maine": (44.693947, -69.381927),
    "Manitoba": (53.760861, -98.813876),
    "Maryland": (39.063946, -76.802101),
    "Massachusetts": (42.230171, -71.530106),
    "Michigan": (43.326618, -84.536095),
    "Minnesota": (45.694454, -93.900192),
    "Mississippi": (32.741646, -89.678696),
    "Missouri": (38.456085, -92.288368),
    "Montana": (46.921925, -110.454353),
    "Nebraska": (41.125370, -98.268082),
    "Nevada": (38.313515, -117.055374),
    "New Brunswick": (46.565316, -66.461916),
    "New Hampshire": (43.452492, -71.563896),
    "New Jersey": (40.298904, -74.521011),
    "New Mexico": (34.840515, -106.248482),
    "New York": (42.165726, -74.948051),
    "Newfoundland and Labrador": (53.135509, -57.660436),
    "North Carolina": (35.630066, -79.806419),
    "North Dakota": (47.528912, -99.784012),
    "Nova Scotia": (44.681987, -63.744311),
    "Ohio": (40.388783, -82.764915),
    "Oklahoma": (35.565342, -96.928917),
    "Ontario": (51.253775, -85.323214),
    "Oregon": (44.572021, -122.070938),
    "Pennsylvania": (40.590752, -77.209755),
    "Prince Edward Island": (46.510712, -63.416813),
    "Quebec": (52.939916, -73.549136),
    "Rhode Island": (41.680893, -71.511780),
    "Saskatchewan": (52.939916, -106.450864),
    "South Carolina": (33.856892, -80.945007),
    "South Dakota": (44.299782, -99.438828),
    "Tennessee": (35.747845, -86.692345),
    "Texas": (31.054487, -97.563461),
    "Utah": (40.150032, -111.862434),
    "Vermont": (44.045876, -72.710686),
    "Virginia": (37.769337, -78.169968),
    "Washington": (47.400902, -121.490494),
    "West Virginia": (38.491226, -80.954453),
    "Wisconsin": (44.268543, -89.616508),
    "Wyoming": (42.755966, -107.302490),
}

# Parameterized 5-factory US manufacturing & distribution network
FACTORIES: dict[str, dict] = {
    "Wicked Choccy's": {
        "city": "Columbus",
        "state": "Georgia",
        "region": "Gulf",
        "lat": 32.4609,
        "lon": -84.9877,
        "primary_division": "Chocolate",
        "max_capacity_share": 0.28,
        "handling_cost_per_unit": 0.05,
    },
    "Lot's O' Nuts": {
        "city": "Wichita",
        "state": "Kansas",
        "region": "Interior",
        "lat": 37.6872,
        "lon": -97.3301,
        "primary_division": "Chocolate",
        "max_capacity_share": 0.30,
        "handling_cost_per_unit": 0.05,
    },
    "Sugar Shack": {
        "city": "Boston",
        "state": "Massachusetts",
        "region": "Atlantic",
        "lat": 42.3601,
        "lon": -71.0589,
        "primary_division": "Sugar",
        "max_capacity_share": 0.30,
        "handling_cost_per_unit": 0.05,
    },
    "The Other Factory": {
        "city": "Portland",
        "state": "Oregon",
        "region": "Pacific",
        "lat": 45.5152,
        "lon": -122.6784,
        "primary_division": "Other",
        "max_capacity_share": 0.26,
        "handling_cost_per_unit": 0.05,
    },
    "Secret Factory": {
        "city": "Salt Lake City",
        "state": "Utah",
        "region": "Pacific",
        "lat": 40.7608,
        "lon": -111.8910,
        "primary_division": "Other",
        "max_capacity_share": 0.24,
        "handling_cost_per_unit": 0.05,
    },
}

# Baseline single-home product-to-factory assignment (reflects centralized legacy routing)
BASELINE_PRODUCT_FACTORY: dict[str, str] = {
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Nutty Crunch Surprise": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Everlasting Gobstopper": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Hair Toffee": "Sugar Shack",
    "Laffy Taffy": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Kazookles": "The Other Factory",
    "Lickable Wallpaper": "The Other Factory",
    "Wonka Gum": "Secret Factory",
}

SHIP_MODE_MULTIPLIERS: dict[str, float] = {
    "Standard Class": 1.00,
    "Second Class": 1.25,
    "First Class": 1.60,
    "Same Day": 2.20,
}

# Average baseline transit days by Ship Mode (for modeled transit savings estimates)
SHIP_MODE_BASE_DAYS: dict[str, float] = {
    "Same Day": 0.04,
    "First Class": 2.18,
    "Second Class": 3.24,
    "Standard Class": 4.98,
}

DEFAULT_RATE_PER_UNIT_MILE = 0.00025
DEFAULT_BASE_HANDLING_PER_UNIT = 0.05


def haversine_miles(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """Compute great-circle distance in statute miles using the Haversine formula."""
    earth_radius_miles = 3958.8
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return earth_radius_miles * c


def get_factory_dataframe() -> pd.DataFrame:
    """Return factory network metadata as a clean DataFrame."""
    rows = []
    for name, meta in FACTORIES.items():
        rows.append({"Factory": name, **meta})
    return pd.DataFrame(rows)


def estimate_freight_cost(
    units: np.ndarray | float,
    distance_miles: np.ndarray | float,
    ship_mode: pd.Series | np.ndarray | str,
    rate_per_unit_mile: float = DEFAULT_RATE_PER_UNIT_MILE,
    base_handling: float = DEFAULT_BASE_HANDLING_PER_UNIT,
) -> np.ndarray | float:
    """Estimate modeled freight & handling cost for a shipment."""
    if isinstance(ship_mode, str):
        mult = SHIP_MODE_MULTIPLIERS.get(ship_mode, 1.0)
    else:
        mult = pd.Series(ship_mode).map(SHIP_MODE_MULTIPLIERS).fillna(1.0).to_numpy()
    return units * (base_handling + rate_per_unit_mile * distance_miles) * mult
