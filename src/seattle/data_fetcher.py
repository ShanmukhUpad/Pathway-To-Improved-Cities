"""
data_fetcher.py — Seattle
Fetches crime and crash data from Seattle Open Data (Socrata) portal.

Crime:  SPD Crime Data: 2008-Present  (tazs-3rd5)
Crash:  SDOT Collisions All Years     (qdnv-25h8)
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
PORTAL   = "https://data.seattle.gov/resource"

CRIME_DATASET_ID = "tazs-3rd5"
CRASH_DATASET_ID = "qdnv-25h8"

CRIME_OUT = os.path.join(SRC_DIR, "crime_monthly_pivot.csv")
CRASH_OUT = os.path.join(SRC_DIR, "traffic_crashes_latest.csv")

CACHE_DAYS = 1
APP_TOKEN  = os.environ.get("SEATTLE_DATA_PORTAL_TOKEN", "")


def is_stale(path: str, days: int = CACHE_DAYS) -> bool:
    if not os.path.exists(path):
        return True
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    return age > timedelta(days=days)


def last_updated(path: str) -> str:
    if not os.path.exists(path):
        return "never"
    return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")


def _fetch_socrata(dataset_id: str, params: dict, chunk_size: int = 50_000) -> pd.DataFrame:
    url     = f"{PORTAL}/{dataset_id}.json"
    headers = {"X-App-Token": APP_TOKEN} if APP_TOKEN else {}
    params  = {**params, "$limit": chunk_size, "$offset": 0}
    frames  = []
    while True:
        resp = requests.get(url, params=params, headers=headers, timeout=120)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        frames.append(pd.DataFrame(batch))
        print(f"  ... {params['$offset'] + len(batch):,} rows fetched")
        if len(batch) < chunk_size:
            break
        params["$offset"] += chunk_size
        time.sleep(0.25)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_crimes(force: bool = False) -> str:
    if not force and not is_stale(CRIME_OUT):
        print(f"[seattle/crimes] Already up to date — {CRIME_OUT}")
        return CRIME_OUT

    print("[seattle/crimes] Fetching from Seattle Open Data …")
    start_year = datetime.now().year - 2

    df = _fetch_socrata(
        CRIME_DATASET_ID,
        params={
            "$where":  f"offense_date >= '{start_year}-01-01T00:00:00'",
            "$select": "neighborhood,offense_date,offense_category",
        },
    )

    if df.empty:
        print("[seattle/crimes] Warning: no data returned.")
        return CRIME_OUT

    df["offense_date"] = pd.to_datetime(df["offense_date"], errors="coerce")
    df["year"]  = df["offense_date"].dt.year
    df["month"] = df["offense_date"].dt.month
    df["community_area"] = df["neighborhood"].str.strip()
    df["primary_type"]   = df["offense_category"].str.strip().str.upper()
    df.dropna(subset=["community_area", "year", "month", "primary_type"], inplace=True)
    df[["year", "month"]] = df[["year", "month"]].astype(int)

    counts = (
        df.groupby(["community_area", "year", "month", "primary_type"])
        .size().reset_index(name="count")
    )
    pivot = counts.pivot_table(
        index=["community_area", "year", "month"],
        columns="primary_type", values="count", fill_value=0,
    ).reset_index()
    pivot.columns.name = None
    pivot.rename(columns={"community_area": "Community Area", "year": "Year", "month": "Month"}, inplace=True)
    pivot.columns = [c if c in ("Community Area", "Year", "Month") else c.upper() for c in pivot.columns]

    pivot.to_csv(CRIME_OUT, index=False)
    print(f"[seattle/crimes] Saved {len(pivot):,} rows → {CRIME_OUT}")
    return CRIME_OUT


def fetch_crashes(force: bool = False) -> str:
    if not force and not is_stale(CRASH_OUT):
        print(f"[seattle/crashes] Already up to date — {CRASH_OUT}")
        return CRASH_OUT

    print("[seattle/crashes] Fetching from Seattle Open Data …")
    start_date = f"{datetime.now().year - 1}-01-01T00:00:00"

    df = _fetch_socrata(
        CRASH_DATASET_ID,
        params={
            "$where":  f"incdttm >= '{start_date}'",
            "$select": "incdttm,weather,lightcond,roadcond,junctiontype,collisiontype,severitydesc,personcount,hitparkedcar,wgs84lat,wgs84long",
        },
    )

    if df.empty:
        print("[seattle/crashes] Warning: no data returned.")
        return CRASH_OUT

    rename = {
        "incdttm":       "CRASH_DATE",
        "weather":       "WEATHER_CONDITION",
        "lightcond":     "LIGHTING_CONDITION",
        "roadcond":      "ROADWAY_SURFACE_COND",
        "junctiontype":  "TRAFFICWAY_TYPE",
        "collisiontype": "FIRST_CRASH_TYPE",
        "severitydesc":  "CRASH_TYPE",
        "personcount":   "NUM_UNITS",
        "hitparkedcar":  "HIT_AND_RUN_I",
        "wgs84lat":      "LATITUDE",
        "wgs84long":     "LONGITUDE",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df.columns = [c.upper() for c in df.columns]

    for col in ["ROAD_DEFECT", "ALIGNMENT", "LANE_CNT", "POSTED_SPEED_LIMIT",
                "TRAFFIC_CONTROL_DEVICE", "DEVICE_CONDITION", "INTERSECTION_RELATED_I",
                "DAMAGE", "BEAT_OF_OCCURRENCE"]:
        if col not in df.columns:
            df[col] = "NOT AVAILABLE"

    df.to_csv(CRASH_OUT, index=False)
    print(f"[seattle/crashes] Saved {len(df):,} rows → {CRASH_OUT}")
    return CRASH_OUT


def refresh_all(force: bool = False):
    fetch_crimes(force=force)
    fetch_crashes(force=force)
    print("[seattle] All datasets refreshed.")


if __name__ == "__main__":
    force = "--force" in sys.argv
    refresh_all(force=force)
