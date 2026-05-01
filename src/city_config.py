"""
city_config.py
--------------
Per-city registry. One source of truth for paths, endpoints, geometry, schema.
All other modules accept a `CityConfig` instead of hardcoding Chicago.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import requests


SRC_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class CityConfig:
    key: str
    name: str
    data_dir: str

    # Per-city CSV filenames (resolved under SRC_DIR/data_dir/)
    crime_csv: str = "crime_monthly_pivot.csv"
    crash_csv: str = "traffic_crashes_latest.csv"
    census_csv: str = "census.csv"

    # Map view
    center: Tuple[float, float] = (0.0, 0.0)
    zoom: int = 10
    lat_bounds: Tuple[float, float] = (-90.0, 90.0)
    lon_bounds: Tuple[float, float] = (-180.0, 180.0)

    # Boundary geometry (Socrata GeoJSON URL OR local file path)
    boundary_url: Optional[str] = None
    boundary_path: Optional[str] = None        # relative to data_dir
    boundary_id_field: str = "GEOID"
    boundary_name_field: str = "NAME"

    # Schema mapping
    crime_area_col: str = "Community Area"
    census_id_col: str = "GEOID"

    # Area-ID semantics: how to normalize CSV `crime_area_col` values so they
    # match boundary `boundary_id_field` keys. "int" (Chicago community areas,
    # NYC precincts), "str" (SF neighborhoods), "upper_str" (LAPD divisions).
    area_id_kind: str = "int"
    # CSV value → boundary ID overrides applied before area_id_kind coercion.
    crime_area_aliases: Tuple[Tuple[str, str], ...] = ()
    # Prefix prepended to area names in UI labels (e.g. "Precinct ").
    area_display_prefix: str = ""
    # Zero-pad str IDs to this width (0 = no padding). E.g. 2 → "5" becomes "05".
    area_id_zfill: int = 0

    # Data Portal (for refresh)
    soda_portal: Optional[str] = None
    crime_dataset_id: Optional[str] = None
    crash_dataset_id: Optional[str] = None
    token_env: str = ""

    # Optional extra Chicago-specific CSVs (empty string = not present)
    crash_csv_legacy: str = ""
    energy_csv: str = ""             # raw energy (space cols) for green_infrastructure
    clean_energy_csv: str = ""       # clean energy (underscore cols) for environment.py
    clean_benchmark_csv: str = ""
    clean_complaints_csv: str = ""
    chives_geojson: str = ""         # ChiVes tract-level equity dataset
    crimes_supplemental_csv: str = ""
    fire_stations_csv: str = ""
    police_stations_csv: str = ""
    acs_csv: str = ""
    population_csv: str = ""
    cta_ridership_csv: str = ""

    # Capability flags
    has_transport_layer: bool = False
    has_energy_layer: bool = False
    has_environment_layer: bool = False

    # Helpers --------------------------------------------------------------
    def path(self, filename: str) -> str:
        return os.path.join(SRC_DIR, self.data_dir, filename)

    @property
    def crime_path(self) -> str:
        return self.path(self.crime_csv)

    @property
    def crash_path(self) -> str:
        return self.path(self.crash_csv)

    @property
    def census_path(self) -> str:
        return self.path(self.census_csv)

    @property
    def crash_legacy_path(self) -> Optional[str]:
        return self.path(self.crash_csv_legacy) if self.crash_csv_legacy else None

    @property
    def energy_path(self) -> Optional[str]:
        return self.path(self.energy_csv) if self.energy_csv else None

    def normalize_area_key(self, value):
        """Normalize a CSV area value to match the boundary id space."""
        import pandas as _pd
        if value is None or (isinstance(value, float) and _pd.isna(value)):
            return None
        alias = dict(self.crime_area_aliases)
        v = alias.get(value, value)
        if isinstance(v, str):
            v = alias.get(v, v)
        kind = self.area_id_kind
        if kind == "int":
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return None
        if kind == "upper_str":
            return str(v).strip().upper()
        s = str(v).strip()
        if self.area_id_zfill > 0:
            s = s.zfill(self.area_id_zfill)
        return s


# ──────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────

CITIES: dict[str, CityConfig] = {
    "chicago": CityConfig(
        key="chicago",
        name="Chicago",
        data_dir=".",  # CSVs live in src/ root
        crime_csv="crime_monthly_pivot.csv",
        crash_csv="traffic_crashes_latest.csv",
        census_csv="censusChicago.csv",
        center=(41.8781, -87.6298),
        zoom=10,
        lat_bounds=(41.6, 42.1),
        lon_bounds=(-87.95, -87.5),
        boundary_url="https://data.cityofchicago.org/resource/igwz-8jzy.geojson",
        boundary_id_field="area_num_1",
        boundary_name_field="community",
        crime_area_col="Community Area",
        census_id_col="Community Area Number",
        soda_portal="https://data.cityofchicago.org/resource",
        crime_dataset_id="ijzp-q8t2",
        crash_dataset_id="85ca-t3if",
        token_env="CHICAGO_DATA_PORTAL_TOKEN",
        has_transport_layer=True,
        has_energy_layer=True,
        has_environment_layer=True,
        crash_csv_legacy="",
        energy_csv="energy_usage_2010.csv",
        clean_energy_csv="clean_energy.csv",
        clean_benchmark_csv="clean_benchmark.csv",
        clean_complaints_csv="clean_complaints.csv",
        chives_geojson="chives-data-public.geojson",
        crimes_supplemental_csv="Crimes_2026.csv",
        fire_stations_csv="Fire_Stations.csv",
        police_stations_csv="Police_Stations.csv",
        acs_csv="acs_5yr.csv",
        population_csv="population_counts.csv",
        cta_ridership_csv="cta_ridership.csv",
    ),
    "new_york": CityConfig(
        key="new_york",
        name="New York",
        data_dir="new_york",
        center=(40.7128, -74.0060),
        zoom=10,
        lat_bounds=(40.4, 41.0),
        lon_bounds=(-74.3, -73.6),
        boundary_url="https://data.cityofnewyork.us/resource/y76i-bdw7.geojson?$limit=200",
        boundary_id_field="precinct",
        boundary_name_field="precinct",
        crime_area_col="Community Area",
        census_id_col="GEOID",
        area_id_kind="int",
        area_display_prefix="Precinct ",
        soda_portal="https://data.cityofnewyork.us/resource",
        crime_dataset_id="qgea-i56i",      # NYPD Complaint Data Historic
        crash_dataset_id="h9gi-nx95",      # Motor Vehicle Collisions - Crashes
        token_env="NYC_DATA_PORTAL_TOKEN",
    ),
    "los_angeles": CityConfig(
        key="los_angeles",
        name="Los Angeles",
        data_dir="los_angeles",
        center=(34.0522, -118.2437),
        zoom=10,
        lat_bounds=(33.7, 34.35),
        lon_bounds=(-118.7, -118.1),
        boundary_path="lapd_divisions.geojson",
        boundary_url=(
            "https://services5.arcgis.com/7nsPwEMP38bSkCjy/arcgis/rest/services/"
            "LAPD_Division/FeatureServer/0/query?where=1=1&outFields=*&f=geojson"
        ),
        boundary_id_field="APREC",
        boundary_name_field="APREC",
        crime_area_col="Community Area",
        census_id_col="GEOID",
        area_id_kind="upper_str",
        crime_area_aliases=(
            ("N Hollywood", "NORTH HOLLYWOOD"),
            ("West LA", "WEST LOS ANGELES"),
        ),
        soda_portal="https://data.lacity.org/resource",
        crime_dataset_id="2nrs-mtv8",
        crash_dataset_id="d5tf-ez2w",
        token_env="LA_DATA_PORTAL_TOKEN",
    ),
    "san_francisco": CityConfig(
        key="san_francisco",
        name="San Francisco",
        data_dir="san_francisco",
        center=(37.7749, -122.4194),
        zoom=11,
        lat_bounds=(37.6, 37.85),
        lon_bounds=(-122.55, -122.35),
        boundary_url="https://data.sfgov.org/resource/ajp5-b2md.geojson",
        boundary_id_field="nhood",
        boundary_name_field="nhood",
        crime_area_col="Community Area",
        census_id_col="GEOID",
        area_id_kind="str",
        crime_area_aliases=(
            ("Financial District/South Beach", "Financial District"),
        ),
        soda_portal="https://data.sfgov.org/resource",
        crime_dataset_id="wg3w-h783",
        crash_dataset_id="ubvf-ztfx",
        token_env="SF_DATA_PORTAL_TOKEN",
    ),
    "philadelphia": CityConfig(
        key="philadelphia",
        name="Philadelphia",
        data_dir="philadelphia",
        center=(39.9526, -75.1652),
        zoom=11,
        lat_bounds=(39.85, 40.15),
        lon_bounds=(-75.3, -74.95),
        boundary_path="police_districts.geojson",
        boundary_id_field="dist_numc",
        boundary_name_field="dist_numc",
        # crime CSV uses "Community Area" header but values = dc_dist (zero-padded)
        crime_area_col="Community Area",
        census_id_col="GEOID",
        area_id_kind="str",
        area_id_zfill=2,           # "5" → "05" to match dist_numc in boundary
        area_display_prefix="District ",
        soda_portal="https://phl.carto.com/api/v2/sql",
        token_env="PHL_DATA_PORTAL_TOKEN",
    ),
}

DEFAULT_CITY_KEY = "chicago"


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def get_city(key: str) -> CityConfig:
    return CITIES[key]


def list_cities() -> list[tuple[str, str]]:
    return [(k, c.name) for k, c in CITIES.items()]


def _simplify_geojson(geo: dict, precision: int = 4) -> dict:
    """Round coordinates to `precision` decimal places.
    Reduces payload size ~50-70% (4 dp ≈ 11m accuracy, fine for city choropleth).
    """
    import copy

    def _round_coords(obj):
        if isinstance(obj, float):
            return round(obj, precision)
        if isinstance(obj, list):
            return [_round_coords(x) for x in obj]
        return obj

    geo = copy.deepcopy(geo)
    for feat in geo.get("features", []):
        geom = feat.get("geometry")
        if geom and "coordinates" in geom:
            geom["coordinates"] = _round_coords(geom["coordinates"])
    return geo


def load_boundary(city: CityConfig) -> tuple[dict, dict]:
    """
    Return (geojson_dict, area_map). Tries local boundary_path first
    (under data_dir), then boundary_url. area_map is {id: name}.
    GeoJSON coordinates are rounded to 4 dp to reduce browser payload.
    """
    geo: Optional[dict] = None

    # Try local file first — but skip empty placeholder files
    if city.boundary_path:
        local = city.path(city.boundary_path)
        if os.path.exists(local):
            with open(local, "r") as f:
                candidate = json.load(f)
            if candidate.get("features"):
                geo = candidate

    # Fallback to remote URL
    if geo is None and city.boundary_url:
        try:
            resp = requests.get(city.boundary_url, timeout=60)
            resp.raise_for_status()
            geo = resp.json()
        except Exception:
            geo = None

    if geo is None:
        raise FileNotFoundError(
            f"No boundary geometry available for {city.name} "
            f"(boundary_path={city.boundary_path}, boundary_url={city.boundary_url})"
        )

    # Reduce coordinate precision to shrink Plotly figure JSON payload
    geo = _simplify_geojson(geo)

    id_field = city.boundary_id_field
    name_field = city.boundary_name_field

    area_map: dict = {}
    for feat in geo.get("features", []):
        props = feat.get("properties", {}) or {}
        raw_id = props.get(id_field)
        if raw_id is None:
            continue
        if city.area_id_kind == "int":
            try:
                key_id = int(float(raw_id))
            except (TypeError, ValueError):
                continue
        elif city.area_id_kind == "upper_str":
            key_id = str(raw_id).strip().upper()
        else:
            key_id = str(raw_id).strip()
        raw_name = props.get(name_field, str(raw_id))
        area_map[key_id] = (
            f"{city.area_display_prefix}{raw_name}" if city.area_display_prefix
            else raw_name
        )

    return geo, area_map
