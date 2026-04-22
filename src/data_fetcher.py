"""
data_fetcher.py
---------------
Fetches the latest datasets from the Chicago Data Portal (Socrata SODA API)
and saves them to local CSV files used by the dashboard.

Run directly:  python data_fetcher.py [--force]
Or import:     from data_fetcher import refresh_all, fetch_crimes, fetch_crashes, is_stale

Optional: set the CHICAGO_DATA_PORTAL_TOKEN environment variable to a free
app token (https://data.cityofchicago.org/profile/app_tokens) to avoid
anonymous rate limits.
"""

import os
import sys
import time
import threading
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load .env from the src/ directory (where this file lives)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PORTAL  = "https://data.cityofchicago.org/resource"

# Chicago Data Portal dataset IDs
CRIME_DATASET_ID = "ijzp-q8t2"  # Crimes - 2001 to Present
CRASH_DATASET_ID = "85ca-t3if"  # Traffic Crashes - Crashes

# Output paths (written into the same src/ directory as the other CSVs)
CRIME_OUT = os.path.join(SRC_DIR, "crime_monthly_pivot.csv")
CRASH_OUT = os.path.join(SRC_DIR, "traffic_crashes_latest.csv")

# Re-fetch if the local file is older than this many days
CACHE_DAYS = 1

# Optional Socrata app token — anonymous access works but is rate-limited
APP_TOKEN = os.environ.get("CHICAGO_DATA_PORTAL_TOKEN", "")

# ── Background refresh state ──────────────────────────────────────────────────

_refresh_done  = threading.Event()   # set() when background refresh completes
_refresh_lock  = threading.Lock()    # prevents overlapping concurrent refreshes


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def is_stale(path: str, days: int = CACHE_DAYS) -> bool:
    """Return True if the file doesn't exist or was last modified more than `days` ago."""
    if not os.path.exists(path):
        return True
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    return age > timedelta(days=days)


def last_updated(path: str) -> str:
    """Human-readable last-modified timestamp, or 'never' if file is missing."""
    if not os.path.exists(path):
        return "never"
    ts = datetime.fromtimestamp(os.path.getmtime(path))
    return ts.strftime("%Y-%m-%d %H:%M")


def _fetch_socrata(dataset_id: str, params: dict, chunk_size: int = 50_000) -> pd.DataFrame:
    """Paginate through a Socrata endpoint and return all rows as a DataFrame."""
    url = f"{PORTAL}/{dataset_id}.json"
    headers = {"X-App-Token": APP_TOKEN} if APP_TOKEN else {}
    params = {**params, "$limit": chunk_size, "$offset": 0}

    frames = []
    while True:
        resp = requests.get(url, params=params, headers=headers, timeout=120)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        frames.append(pd.DataFrame(batch))
        total_so_far = params["$offset"] + len(batch)
        print(f"  ... {total_so_far:,} rows fetched")
        if len(batch) < chunk_size:
            break
        params["$offset"] += chunk_size
        time.sleep(0.25)   # be polite to the API

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ──────────────────────────────────────────────
# Crime data
# ──────────────────────────────────────────────

def fetch_crimes(force: bool = False) -> str:
    """
    Download recent Chicago crime records from the Data Portal and rebuild
    crime_monthly_pivot.csv (Community Area × Year × Month pivot by crime type).

    Parameters
    ----------
    force : bool
        If True, re-download even when the local file is fresh.

    Returns
    -------
    str
        Absolute path to the output CSV.
    """
    if not force and not is_stale(CRIME_OUT):
        print(f"[crimes] Already up to date — {CRIME_OUT}")
        return CRIME_OUT

    print("[crimes] Fetching from Chicago Data Portal (this may take a minute)...")
    start_year = datetime.now().year - 2  # last ~3 years of data

    df = _fetch_socrata(
        CRIME_DATASET_ID,
        params={
            "$where":  f"year >= '{start_year}'",
            "$select": "community_area,year,date,primary_type",
        },
    )

    if df.empty:
        print("[crimes] Warning: no data returned from the API.")
        return CRIME_OUT

    # Clean & type-cast
    df["year"]           = pd.to_numeric(df["year"],           errors="coerce")
    df["date"]           = pd.to_datetime(df["date"],          errors="coerce")
    df["month"]          = df["date"].dt.month
    df["community_area"] = pd.to_numeric(df["community_area"], errors="coerce")
    df.dropna(subset=["community_area", "year", "month", "primary_type"], inplace=True)
    df[["community_area", "year", "month"]] = (
        df[["community_area", "year", "month"]].astype(int)
    )

    # Count crimes per (community area, year, month, primary type)
    counts = (
        df.groupby(["community_area", "year", "month", "primary_type"])
        .size()
        .reset_index(name="count")
    )

    # Pivot primary_type → columns, fill missing combos with 0
    pivot = counts.pivot_table(
        index=["community_area", "year", "month"],
        columns="primary_type",
        values="count",
        fill_value=0,
    ).reset_index()
    pivot.columns.name = None

    # Rename to match the column names expected by public_safety.py
    pivot.rename(columns={
        "community_area": "Community Area",
        "year":           "Year",
        "month":          "Month",
    }, inplace=True)

    # Uppercase crime-type column names (ARSON, ASSAULT, BATTERY, …)
    pivot.columns = [
        c if c in ("Community Area", "Year", "Month") else c.upper()
        for c in pivot.columns
    ]

    pivot.to_csv(CRIME_OUT, index=False)
    print(f"[crimes] Saved {len(pivot):,} rows → {CRIME_OUT}")
    return CRIME_OUT


# ──────────────────────────────────────────────
# Traffic crash data
# ──────────────────────────────────────────────

def fetch_crashes(force: bool = False) -> str:
    """
    Download recent Chicago traffic crash records from the Data Portal and
    save to traffic_crashes_latest.csv.

    Parameters
    ----------
    force : bool
        If True, re-download even when the local file is fresh.

    Returns
    -------
    str
        Absolute path to the output CSV.
    """
    if not force and not is_stale(CRASH_OUT):
        print(f"[crashes] Already up to date — {CRASH_OUT}")
        return CRASH_OUT

    print("[crashes] Fetching from Chicago Data Portal (this may take a minute)...")
    start_date = f"{datetime.now().year - 1}-01-01T00:00:00"

    df = _fetch_socrata(
        CRASH_DATASET_ID,
        params={
            "$where":  f"crash_date >= '{start_date}'",
            "$select": (
                "crash_date,weather_condition,lighting_condition,"
                "roadway_surface_cond,road_defect,alignment,trafficway_type,"
                "lane_cnt,posted_speed_limit,traffic_control_device,"
                "device_condition,intersection_related_i,first_crash_type,"
                "crash_type,damage,num_units,hit_and_run_i,"
                "latitude,longitude,beat_of_occurrence"
            ),
        },
    )

    if df.empty:
        print("[crashes] Warning: no data returned from the API.")
        return CRASH_OUT

    # Uppercase column names to match crash.py expectations
    df.columns = [c.upper() for c in df.columns]

    df.to_csv(CRASH_OUT, index=False)
    print(f"[crashes] Saved {len(df):,} rows → {CRASH_OUT}")
    return CRASH_OUT


# ──────────────────────────────────────────────
# Convenience
# ──────────────────────────────────────────────

def refresh_all(force: bool = False):
    """Refresh all datasets from the Chicago Data Portal."""
    fetch_crimes(force=force)
    fetch_crashes(force=force)
    print("[done] All datasets refreshed.")


# ── Scheduled / startup refresh ───────────────────────────────────────────────

def start_background_refresh():
    """
    Spawn a daemon thread to refresh stale datasets on startup.
    No-op if data is already fresh. Safe to call on every Streamlit rerun —
    _refresh_lock prevents overlapping runs.

    The thread only performs disk I/O. All st.* calls must happen on the
    render thread after checking _refresh_done.is_set().
    """
    def _worker():
        if not _refresh_lock.acquire(blocking=False):
            return   # another refresh is already in progress
        try:
            refresh_all(force=False)
            _refresh_done.set()
        finally:
            _refresh_lock.release()

    if is_stale(CRIME_OUT) or is_stale(CRASH_OUT):
        threading.Thread(target=_worker, daemon=True).start()


def start_scheduler():
    """
    Start an APScheduler BackgroundScheduler that triggers refresh_all()
    every day at 06:00. Intended for production deployments.
    Use st.cache_resource to ensure only one scheduler per server process.
    """
    from apscheduler.schedulers.background import BackgroundScheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        refresh_all,
        trigger="cron",
        hour=6,
        minute=0,
        kwargs={"force": True},
        id="daily_refresh",
        replace_existing=True,
    )
    scheduler.start()
    return scheduler


if __name__ == "__main__":
    force = "--force" in sys.argv
    refresh_all(force=force)
