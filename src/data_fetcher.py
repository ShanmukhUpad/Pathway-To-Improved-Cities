"""
data_fetcher.py
---------------
Fetches the latest datasets from city open-data portals (Socrata SODA APIs)
and saves them to per-city CSV files used by the dashboard.

Each city's connection details (portal URL, dataset IDs, output dir, token env)
live in `city_config.CITIES`. Multi-city refreshes run in a thread pool.

Run directly:  python data_fetcher.py [--force] [--city chicago|new_york|...]
"""

import argparse
import os
import sys
import time
<<<<<<< Updated upstream
import requests
import pandas as pd
=======
import threading
from concurrent.futures import ThreadPoolExecutor
>>>>>>> Stashed changes
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

from city_config import CITIES, CityConfig, get_city, SRC_DIR

# Load .env from the src/ directory
load_dotenv(os.path.join(SRC_DIR, ".env"))

# Re-fetch if local file older than this many days
CACHE_DAYS = 1

<<<<<<< Updated upstream
# Optional Socrata app token — anonymous access works but is rate-limited
APP_TOKEN = os.environ.get("CHICAGO_DATA_PORTAL_TOKEN", "")
=======
# Background refresh state
_refresh_done = threading.Event()
_refresh_lock = threading.Lock()
>>>>>>> Stashed changes


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def is_stale(path: str, days: int = CACHE_DAYS) -> bool:
    if not os.path.exists(path):
        return True
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    return age > timedelta(days=days)


def last_updated(path: str) -> str:
    if not os.path.exists(path):
        return "never"
    return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _fetch_socrata(portal: str, dataset_id: str, params: dict,
                   token: str = "", chunk_size: int = 50_000) -> pd.DataFrame:
    """Paginate a Socrata endpoint and return all rows."""
    url = f"{portal}/{dataset_id}.json"
    headers = {"X-App-Token": token} if token else {}
    params = {**params, "$limit": chunk_size, "$offset": 0}

    frames = []
    while True:
        resp = requests.get(url, params=params, headers=headers, timeout=120)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        frames.append(pd.DataFrame(batch))
        total = params["$offset"] + len(batch)
        print(f"  ... {total:,} rows fetched ({dataset_id})")
        if len(batch) < chunk_size:
            break
        params["$offset"] += chunk_size
        time.sleep(0.25)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ──────────────────────────────────────────────
# Chicago crime ETL (only Chicago has a known transform recipe)
# ──────────────────────────────────────────────

def _fetch_chicago_crimes(city: CityConfig, force: bool) -> str:
    out = city.crime_path
    if not force and not is_stale(out):
        print(f"[{city.key}/crimes] up to date — {out}")
        return out

    print(f"[{city.key}/crimes] fetching...")
    start_year = datetime.now().year - 2
    token = os.environ.get(city.token_env, "")

    df = _fetch_socrata(
        city.soda_portal, city.crime_dataset_id,
        params={
            "$where":  f"year >= '{start_year}'",
            "$select": "community_area,year,date,primary_type",
        },
        token=token,
    )
    if df.empty:
        print(f"[{city.key}/crimes] no rows")
        return out

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.month
    df["community_area"] = pd.to_numeric(df["community_area"], errors="coerce")
    df.dropna(subset=["community_area", "year", "month", "primary_type"], inplace=True)
    df[["community_area", "year", "month"]] = df[["community_area", "year", "month"]].astype(int)

    counts = (df.groupby(["community_area", "year", "month", "primary_type"])
                .size().reset_index(name="count"))
    pivot = counts.pivot_table(
        index=["community_area", "year", "month"],
        columns="primary_type", values="count", fill_value=0,
    ).reset_index()
    pivot.columns.name = None
    pivot.rename(columns={
        "community_area": "Community Area",
        "year": "Year",
        "month": "Month",
    }, inplace=True)
    pivot.columns = [c if c in ("Community Area", "Year", "Month") else c.upper()
                     for c in pivot.columns]

    _ensure_dir(out)
    pivot.to_csv(out, index=False)
    print(f"[{city.key}/crimes] saved {len(pivot):,} rows → {out}")
    return out


# ──────────────────────────────────────────────
# Chicago crash ETL
# ──────────────────────────────────────────────

def _fetch_chicago_crashes(city: CityConfig, force: bool) -> str:
    out = city.crash_path
    if not force and not is_stale(out):
        print(f"[{city.key}/crashes] up to date — {out}")
        return out

    print(f"[{city.key}/crashes] fetching...")
    start = f"{datetime.now().year - 1}-01-01T00:00:00"
    token = os.environ.get(city.token_env, "")

    df = _fetch_socrata(
        city.soda_portal, city.crash_dataset_id,
        params={
            "$where":  f"crash_date >= '{start}'",
            "$select": (
                "crash_date,weather_condition,lighting_condition,"
                "roadway_surface_cond,road_defect,alignment,trafficway_type,"
                "lane_cnt,posted_speed_limit,traffic_control_device,"
                "device_condition,intersection_related_i,first_crash_type,"
                "crash_type,damage,num_units,hit_and_run_i,"
                "latitude,longitude,beat_of_occurrence"
            ),
        },
        token=token,
    )
    if df.empty:
        print(f"[{city.key}/crashes] no rows")
        return out

    df.columns = [c.upper() for c in df.columns]
    _ensure_dir(out)
    df.to_csv(out, index=False)
    print(f"[{city.key}/crashes] saved {len(df):,} rows → {out}")
    return out


# ──────────────────────────────────────────────
# Per-city dispatch
# ──────────────────────────────────────────────

# City-specific ETL recipes. Add entries here when wiring NYC/LA/SF refresh.
_CRIME_FETCHERS = {
    "chicago": _fetch_chicago_crimes,
}
_CRASH_FETCHERS = {
    "chicago": _fetch_chicago_crashes,
}


<<<<<<< Updated upstream
=======
def fetch_crimes(city: CityConfig, force: bool = False) -> str:
    fn = _CRIME_FETCHERS.get(city.key)
    if fn is None:
        # No remote ETL — rely on the prebuilt CSV that ships with the repo
        return city.crime_path
    return fn(city, force)


def fetch_crashes(city: CityConfig, force: bool = False) -> str:
    fn = _CRASH_FETCHERS.get(city.key)
    if fn is None:
        return city.crash_path
    return fn(city, force)


def refresh_city(city_key: str, force: bool = False) -> None:
    city = get_city(city_key)
    fetch_crimes(city, force=force)
    fetch_crashes(city, force=force)


def refresh_all_cities(force: bool = False) -> None:
    """Refresh all cities in parallel (one thread per city, IO-bound)."""
    with ThreadPoolExecutor(max_workers=len(CITIES)) as pool:
        list(pool.map(lambda k: refresh_city(k, force=force), CITIES.keys()))
    print("[done] all cities refreshed")


# ──────────────────────────────────────────────
# Backwards-compat shims (deprecated; kept so legacy callers don't break)
# ──────────────────────────────────────────────

CRIME_OUT = get_city("chicago").crime_path
CRASH_OUT = get_city("chicago").crash_path


def refresh_all(force: bool = False) -> None:
    refresh_all_cities(force=force)


# ──────────────────────────────────────────────
# Background refresh + scheduler
# ──────────────────────────────────────────────

def _any_stale() -> bool:
    for city in CITIES.values():
        if is_stale(city.crime_path) or is_stale(city.crash_path):
            return True
    return False


def start_background_refresh() -> None:
    """Spawn daemon thread to refresh stale datasets across all cities."""
    def _worker():
        if not _refresh_lock.acquire(blocking=False):
            return
        try:
            refresh_all_cities(force=False)
            _refresh_done.set()
        finally:
            _refresh_lock.release()

    if _any_stale():
        threading.Thread(target=_worker, daemon=True).start()


def start_scheduler():
    """Daily 06:00 refresh across all cities."""
    from apscheduler.schedulers.background import BackgroundScheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        refresh_all_cities,
        trigger="cron", hour=6, minute=0,
        kwargs={"force": True},
        id="daily_refresh", replace_existing=True,
    )
    scheduler.start()
    return scheduler


>>>>>>> Stashed changes
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--city", choices=list(CITIES.keys()),
                   help="Refresh a single city (default: all)")
    args = p.parse_args()

    if args.city:
        refresh_city(args.city, force=args.force)
    else:
        refresh_all_cities(force=args.force)
