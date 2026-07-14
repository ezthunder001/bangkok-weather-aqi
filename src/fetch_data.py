"""Bangkok weather + air-quality data pipeline.

Primary source: Open-Meteo (free, no API key)
  - Historical weather:  https://archive-api.open-meteo.com/v1/archive
  - Air quality (CAMS):  https://air-quality-api.open-meteo.com/v1/air-quality

The fetcher is provider-agnostic at the DataFrame level, so an
OpenWeatherMap (or IQAir / Thai PCD) backend can be added later without
touching the dashboard.

Output: data/bangkok_daily.csv — one row per day with weather, air
quality, and derived RC-research metrics (outdoor test validity index).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Windows cp1252 console crashes on unicode glyphs — force UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

BANGKOK_LAT, BANGKOK_LON = 13.7563, 100.5018
TIMEZONE = "Asia/Bangkok"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DAILY_CSV = DATA_DIR / "bangkok_daily.csv"

WEATHER_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "shortwave_radiation_sum",  # MJ/m²/day
    "cloud_cover_mean",
    "wind_speed_10m_max",
]

AQ_HOURLY_VARS = ["pm2_5", "pm10", "ozone", "us_aqi"]

# Open-Meteo air-quality archive (CAMS) starts mid-2022
AQ_EARLIEST = date(2022, 8, 1)


def _get(url: str, params: dict, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries - 1:
                raise
            wait = 5 * (attempt + 1)
            print(f"  retry {attempt + 1} after error: {exc} (waiting {wait}s)")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def fetch_weather(start: date, end: date) -> pd.DataFrame:
    """Daily weather from the Open-Meteo historical archive."""
    print(f"Fetching weather {start} → {end} ...")
    payload = _get(
        ARCHIVE_URL,
        {
            "latitude": BANGKOK_LAT,
            "longitude": BANGKOK_LON,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": ",".join(WEATHER_DAILY_VARS),
            "timezone": TIMEZONE,
        },
    )
    df = pd.DataFrame(payload["daily"])
    df["date"] = pd.to_datetime(df.pop("time"))
    print(f"  {len(df)} days of weather")
    return df


def fetch_air_quality(start: date, end: date) -> pd.DataFrame:
    """Hourly CAMS air quality, aggregated to daily mean/max."""
    start = max(start, AQ_EARLIEST)
    frames = []
    chunk_start = start
    while chunk_start <= end:  # one calendar year per request
        chunk_end = min(date(chunk_start.year, 12, 31), end)
        print(f"Fetching air quality {chunk_start} → {chunk_end} ...")
        payload = _get(
            AIR_QUALITY_URL,
            {
                "latitude": BANGKOK_LAT,
                "longitude": BANGKOK_LON,
                "start_date": chunk_start.isoformat(),
                "end_date": chunk_end.isoformat(),
                "hourly": ",".join(AQ_HOURLY_VARS),
                "timezone": TIMEZONE,
            },
        )
        frames.append(pd.DataFrame(payload["hourly"]))
        chunk_start = date(chunk_start.year + 1, 1, 1)

    hourly = pd.concat(frames, ignore_index=True)
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly["date"] = hourly["time"].dt.normalize()

    daily = hourly.groupby("date").agg(
        pm2_5=("pm2_5", "mean"),
        pm2_5_max=("pm2_5", "max"),
        pm10=("pm10", "mean"),
        ozone=("ozone", "mean"),
        us_aqi=("us_aqi", "mean"),
        us_aqi_max=("us_aqi", "max"),
    ).reset_index()
    daily = daily.dropna(subset=["pm2_5"])
    print(f"  {len(daily)} days of air quality")
    return daily


# ---------------------------------------------------------------- metrics

THAI_SEASONS = {  # Thai Meteorological Department convention
    "Cool": (11, 12, 1, 2),
    "Hot": (3, 4, 5),
    "Rainy": (6, 7, 8, 9, 10),
}


def tag_season(month: int) -> str:
    for season, months in THAI_SEASONS.items():
        if month in months:
            return season
    return "?"


def rc_test_validity(row: pd.Series) -> float:
    """0–100 score: how valid is this day for outdoor RC coating testing.

    Radiative-cooling field tests in Bangkok need clear sky (solar load +
    unobstructed 8–13 µm window), low haze (PM2.5 scatters both solar and
    LWIR), and no rain. Weights: solar 40, haze 30, cloud 30; rain >1 mm
    collapses the score (wet samples invalidate ΔT measurement).
    """
    if pd.isna(row.get("pm2_5")):
        return np.nan
    solar = np.clip((row["shortwave_radiation_sum"] - 10) / (22 - 10), 0, 1)
    haze = np.clip((75 - row["pm2_5"]) / (75 - 15), 0, 1)
    sky = np.clip((85 - row["cloud_cover_mean"]) / (85 - 25), 0, 1)
    score = 100 * (0.40 * solar + 0.30 * haze + 0.30 * sky)
    if row["precipitation_sum"] > 1.0:
        score *= 0.15
    return round(score, 1)


def classify_validity(score: float) -> str:
    if pd.isna(score):
        return "NO_DATA"
    if score >= 70:
        return "VALID"
    if score >= 40:
        return "MARGINAL"
    return "INVALID"


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map(tag_season)
    df["test_validity"] = df.apply(rc_test_validity, axis=1)
    df["test_class"] = df["test_validity"].map(classify_validity)
    return df


def build_dataset(start: date, end: date) -> pd.DataFrame:
    weather = fetch_weather(start, end)
    air = fetch_air_quality(start, end)
    merged = weather.merge(air, on="date", how="left")
    merged = add_derived(merged)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--force", action="store_true", help="refetch even if cache is fresh")
    args = parser.parse_args()

    end = date.today() - timedelta(days=3)  # archive API lags a few days
    start = date.fromisoformat(args.start)

    if DAILY_CSV.exists() and not args.force:
        cached = pd.read_csv(DAILY_CSV, parse_dates=["date"])
        if cached["date"].max().date() >= end - timedelta(days=2):
            print(f"Cache is fresh ({DAILY_CSV}, {len(cached)} rows) — use --force to refetch")
            return

    df = build_dataset(start, end)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DAILY_CSV.with_suffix(".csv.tmp")  # atomic write
    df.to_csv(tmp, index=False)
    tmp.replace(DAILY_CSV)

    print(f"\nSaved {len(df)} rows → {DAILY_CSV}")
    aq_days = df["pm2_5"].notna().sum()
    print(f"Weather coverage: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Air-quality coverage: {aq_days} days (from {AQ_EARLIEST})")


if __name__ == "__main__":
    main()
