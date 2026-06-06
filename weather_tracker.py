#!/usr/bin/env python3
"""
Weather Forecast Tracker & Accuracy Analyser
=============================================
Collects hourly forecasts (up to 5 days ahead) from all Windguru models
for spot 56996 (Bielersee, Nidau), records actual measurements from
WSA-Ipsach (wsa-ipsach.meteobase.ch) every hour, then compares which
model was most accurate.

Run once per hour via cron:
    0 * * * * /usr/bin/python3 /path/to/weather_tracker.py

Or run the analysis any time:
    python3 weather_tracker.py --analyse

Dependencies (install once):
    pip install requests playwright pandas tabulate
    playwright install chromium
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import pandas as pd
from tabulate import tabulate

# ── optional browser import (only needed for Windguru scraping) ─────────────
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# ── configuration ────────────────────────────────────────────────────────────

WINDGURU_SPOT_ID = 56996
WINDGURU_URL     = f"https://www.windguru.cz/{WINDGURU_SPOT_ID}"

# WSA-Ipsach station coordinates (Lake Biel, close to Windguru spot)
STATION_LAT = 47.113
STATION_LON = 7.224

# All Windguru model slugs that typically appear for Swiss inland spots.
# The scraper will discover whatever models are actually shown on the page.
KNOWN_MODELS = [
    "WG", "GFS", "GFS HD", "ECMWF", "ICON", "AROME", "HARMONIE",
    "ICON-D2", "GDPS", "NAM"
]

DATA_DIR = Path("weather_data")
FORECASTS_DIR = DATA_DIR / "forecasts"
MEASUREMENTS_DIR = DATA_DIR / "measurements"
ANALYSIS_DIR = DATA_DIR / "analysis"

for d in [FORECASTS_DIR, MEASUREMENTS_DIR, ANALYSIS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(DATA_DIR / "tracker.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────────────

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def hour_key(dt: datetime) -> str:
    """YYYY-MM-DDTHH (UTC, hour-truncated) – used as the primary time key."""
    return dt.strftime("%Y-%m-%dT%H")

def forecast_file(run_time: datetime) -> Path:
    return FORECASTS_DIR / f"{hour_key(run_time)}.json"

def measurement_file(obs_time: datetime) -> Path:
    return MEASUREMENTS_DIR / f"{hour_key(obs_time)}.json"

# ── 1. Windguru forecast scraping ────────────────────────────────────────────

def _windguru_internal_api(spot_id: int) -> dict | None:
    """
    Attempt to hit Windguru's internal forecast endpoint directly.
    Returns parsed JSON or None if it fails.
    """
    # Windguru uses a PHP API that is called client-side.
    # The salt/hash changes per session, so we first fetch the page to get them.
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Referer": f"https://www.windguru.cz/{spot_id}",
    })
    try:
        page_resp = session.get(f"https://www.windguru.cz/{spot_id}", timeout=15)
        page_resp.raise_for_status()

        # Extract the 'uid' / salt used in the hash
        uid_match = re.search(r'"uid"\s*:\s*"([^"]+)"', page_resp.text)
        salt = uid_match.group(1) if uid_match else str(int(time.time()))

        import hashlib
        hash_val = hashlib.md5(f"{salt}{spot_id}".encode()).hexdigest()

        params = {
            "q":           "forecast",
            "id_spot":     spot_id,
            "uid":         salt,
            "hash":        hash_val,
            "lang":        "en",
            "no_wave":     0,
            "WGCACHEABLE": 21600,
        }
        api_resp = session.get(
            "https://www.windguru.cz/int/iapi.php",
            params=params,
            timeout=20,
        )
        api_resp.raise_for_status()
        data = api_resp.json()
        return data
    except Exception as exc:
        log.warning("Internal API call failed: %s", exc)
        return None


def _windguru_playwright(spot_id: int) -> dict | None:
    """
    Fall-back: launch a headless browser, wait for Windguru to render,
    then intercept the forecast XHR to get the raw JSON payload.
    """
    if not PLAYWRIGHT_AVAILABLE:
        log.error("Playwright not installed. Run:  pip install playwright && playwright install chromium")
        return None

    captured: dict | None = None

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx     = browser.new_context()
        page    = ctx.new_page()

        def handle_response(response):
            nonlocal captured
            if "iapi.php" in response.url and "forecast" in response.url:
                try:
                    captured = response.json()
                except Exception:
                    pass

        page.on("response", handle_response)
        page.goto(f"https://www.windguru.cz/{spot_id}", timeout=30_000)
        page.wait_for_timeout(8_000)   # let JS load all models
        browser.close()

    return captured


def fetch_windguru_forecasts(spot_id: int = WINDGURU_SPOT_ID) -> dict | None:
    """
    Returns the raw Windguru forecast JSON (all models) or None on failure.
    Tries the lightweight API first, then falls back to headless browser.
    """
    log.info("Fetching Windguru forecasts for spot %s …", spot_id)
    data = _windguru_internal_api(spot_id)
    if data and "fcst" in data:
        log.info("  ✓ Got forecast via internal API (%d models)", len(data["fcst"]))
        return data
    log.info("  ↳ Internal API did not return forecast; trying Playwright …")
    data = _windguru_playwright(spot_id)
    if data and "fcst" in data:
        log.info("  ✓ Got forecast via Playwright (%d models)", len(data["fcst"]))
        return data
    log.error("  ✗ Could not retrieve Windguru forecasts.")
    return None


def parse_windguru_forecasts(raw: dict) -> list[dict]:
    """
    Convert the raw Windguru JSON into a flat list of hourly records:
        {model, valid_time_utc, hour_key, wind_speed_kn, wind_gust_kn,
         wind_dir_deg, temp_c, precip_mm, lead_hours}
    lead_hours = how many hours ahead the forecast is from the model run time.
    """
    records = []
    fcst_block = raw.get("fcst", {})

    # Model run offset: Windguru stores forecast hours as offsets from
    # a reference timestamp per model.
    for model_id, model_data in fcst_block.items():
        model_name = model_data.get("model_name", model_id)
        timestamps = model_data.get("fcst_hour_idx", [])  # epoch seconds per step
        hours      = model_data.get("hours", [])          # hour-of-day (optional)

        wind_speed = model_data.get("WINDSPD", [])
        wind_gust  = model_data.get("GUST",    [])
        wind_dir   = model_data.get("WINDDIR", [])
        temp       = model_data.get("TMP",     [])
        precip     = model_data.get("APCP",    [])

        # init_timestamp: when was this model run initialised
        init_ts = model_data.get("init_d", None)
        try:
            init_dt = datetime.fromtimestamp(float(init_ts), tz=timezone.utc)
        except (TypeError, ValueError):
            init_dt = utc_now().replace(minute=0, second=0, microsecond=0)

        for i, ts in enumerate(timestamps):
            try:
                valid_dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            except (TypeError, ValueError):
                continue

            lead = int((valid_dt - init_dt).total_seconds() / 3600)
            if lead > 5 * 24:
                continue  # beyond 5-day window

            records.append({
                "model":          model_name,
                "init_time_utc":  init_dt.isoformat(),
                "valid_time_utc": valid_dt.isoformat(),
                "hour_key":       hour_key(valid_dt),
                "lead_hours":     lead,
                "wind_speed_kn":  _safe(wind_speed, i),
                "wind_gust_kn":   _safe(wind_gust, i),
                "wind_dir_deg":   _safe(wind_dir, i),
                "temp_c":         _safe(temp, i),
                "precip_mm":      _safe(precip, i),
            })

    log.info("  Parsed %d hourly forecast records across %d models.",
             len(records), len(fcst_block))
    return records


def _safe(lst, i):
    try:
        v = lst[i]
        return float(v) if v is not None else None
    except (IndexError, TypeError, ValueError):
        return None


def save_forecasts(records: list[dict], run_time: datetime) -> None:
    fp = forecast_file(run_time)
    payload = {
        "fetch_time_utc": run_time.isoformat(),
        "spot_id":        WINDGURU_SPOT_ID,
        "records":        records,
    }
    fp.write_text(json.dumps(payload, indent=2))
    log.info("  Saved %d forecast records → %s", len(records), fp)


# ── 2. WSA-Ipsach measurements ───────────────────────────────────────────────

def fetch_meteobase_measurement() -> dict | None:
    """
    The wsa-ipsach.meteobase.ch site is powered by meteoBase.ch.
    It exposes a lightweight PHP endpoint that returns current sensor data.
    We try two known URL patterns.  If both fail we fall back to the
    open MeteoSwiss data API (data.geo.admin.ch) for the nearest station.
    """
    log.info("Fetching WSA-Ipsach measurement …")

    # ── attempt 1: meteoBase JSON endpoint ──────────────────────────────────
    for url in [
        "https://wsa-ipsach.meteobase.ch/api/data.php?format=json",
        "https://wsa-ipsach.meteobase.ch/data.php?format=json",
        "https://wsa-ipsach.meteobase.ch/?format=json",
    ]:
        try:
            r = requests.get(url, timeout=10,
                             headers={"User-Agent": "WeatherTracker/1.0"})
            if r.status_code == 200 and r.content:
                data = r.json()
                parsed = _parse_meteobase_json(data)
                if parsed:
                    log.info("  ✓ Got meteoBase measurement via %s", url)
                    return parsed
        except Exception as exc:
            log.debug("  meteoBase attempt failed (%s): %s", url, exc)

    # ── attempt 2: meteoBase HTML scrape (lightweight) ───────────────────────
    parsed = _scrape_meteobase_html()
    if parsed:
        return parsed

    # ── attempt 3: MeteoSwiss open data API (nearest station BIE/Biel) ───────
    log.info("  ↳ Falling back to MeteoSwiss open data for Biel/Bienne …")
    parsed = _fetch_meteoswiss_opendata()
    if parsed:
        return parsed

    log.error("  ✗ Could not retrieve any measurement.")
    return None


def _parse_meteobase_json(data: dict) -> dict | None:
    """Extract wind / temp / precip from meteoBase JSON (format may vary)."""
    try:
        now = utc_now()
        return {
            "source":        "wsa-ipsach.meteobase.ch",
            "obs_time_utc":  now.isoformat(),
            "hour_key":      hour_key(now),
            "wind_speed_kn": _to_knots_from_kmh(data.get("wind_speed") or data.get("ws")),
            "wind_gust_kn":  _to_knots_from_kmh(data.get("wind_gust")  or data.get("wg")),
            "wind_dir_deg":  _float(data.get("wind_dir") or data.get("wd")),
            "temp_c":        _float(data.get("temp") or data.get("temperature") or data.get("ta")),
            "precip_mm":     _float(data.get("precip") or data.get("rain") or data.get("rr")),
        }
    except Exception:
        return None


def _scrape_meteobase_html() -> dict | None:
    """
    Parse the meteoBase.ch station page HTML to extract current readings.
    meteoBase pages embed sensor values in <span id="val_..."> tags.
    """
    try:
        r = requests.get(
            "https://wsa-ipsach.meteobase.ch/",
            timeout=12,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; WeatherTracker/1.0)",
                "Accept-Language": "de,en;q=0.9",
            },
            allow_redirects=True,
        )
        html = r.text

        def extract(pattern):
            m = re.search(pattern, html, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1).replace(",", "."))
                except ValueError:
                    pass
            return None

        # Common meteoBase.ch patterns
        wind_kmh  = (
            extract(r'id=["\']val_windspeed["\'][^>]*>([\d.,]+)')
            or extract(r'Wind.*?<[^>]+>([\d.,]+)\s*km/h')
        )
        gust_kmh  = (
            extract(r'id=["\']val_windgust["\'][^>]*>([\d.,]+)')
            or extract(r'B.+?<[^>]+>([\d.,]+)\s*km/h')
        )
        wind_dir  = extract(r'id=["\']val_winddir["\'][^>]*>([\d.,]+)')
        temp_c    = (
            extract(r'id=["\']val_temperature["\'][^>]*>([\d.,\-]+)')
            or extract(r'Temp[^<]*<[^>]+>([\d.,\-]+)\s*°')
        )
        precip_mm = (
            extract(r'id=["\']val_rain["\'][^>]*>([\d.,]+)')
            or extract(r'Regen[^<]*<[^>]+>([\d.,]+)\s*mm')
        )

        if wind_kmh is not None or temp_c is not None:
            now = utc_now()
            result = {
                "source":        "wsa-ipsach.meteobase.ch (html)",
                "obs_time_utc":  now.isoformat(),
                "hour_key":      hour_key(now),
                "wind_speed_kn": _to_knots_from_kmh(wind_kmh),
                "wind_gust_kn":  _to_knots_from_kmh(gust_kmh),
                "wind_dir_deg":  wind_dir,
                "temp_c":        temp_c,
                "precip_mm":     precip_mm,
            }
            log.info("  ✓ Got meteoBase measurement via HTML scrape")
            return result
    except Exception as exc:
        log.debug("  HTML scrape failed: %s", exc)
    return None


def _fetch_meteoswiss_opendata() -> dict | None:
    """
    Use the MeteoSwiss / geo.admin.ch open data JSON feeds.
    Finds Biel/Bienne (station BIE) or the nearest station to WSA-Ipsach.
    """
    # Station abbreviation for Biel/Bienne known to MeteoSwiss: BIE
    TARGET_STATION = "BIE"
    BASE = "https://data.geo.admin.ch"

    variable_urls = {
        "wind_speed_kmh": (
            f"{BASE}/ch.meteoschweiz.messwerte-windgeschwindigkeit-kmh-10min"
            f"/ch.meteoschweiz.messwerte-windgeschwindigkeit-kmh-10min_en.json"
        ),
        "wind_gust_kmh": (
            f"{BASE}/ch.meteoschweiz.messwerte-wind-boeenspitze-kmh-10min"
            f"/ch.meteoschweiz.messwerte-wind-boeenspitze-kmh-10min_en.json"
        ),
        "wind_dir": (
            f"{BASE}/ch.meteoschweiz.messwerte-windrichtung-10min"
            f"/ch.meteoschweiz.messwerte-windrichtung-10min_en.json"
        ),
        "temp_c": (
            f"{BASE}/ch.meteoschweiz.messwerte-lufttemperatur-10min"
            f"/ch.meteoschweiz.messwerte-lufttemperatur-10min_en.json"
        ),
        "precip_mm": (
            f"{BASE}/ch.meteoschweiz.messwerte-niederschlag-10min"
            f"/ch.meteoschweiz.messwerte-niederschlag-10min_en.json"
        ),
    }

    readings = {}
    obs_time = None

    for var, url in variable_urls.items():
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            geo = r.json()
            for feature in geo.get("features", []):
                props = feature.get("properties", {})
                station_id = props.get("station_id", "") or props.get("nat_abbr", "")
                if station_id.upper() == TARGET_STATION:
                    val = props.get("value")
                    readings[var] = _float(val)
                    if obs_time is None:
                        obs_time = props.get("reference_ts") or props.get("date")
                    break
        except Exception as exc:
            log.debug("  MeteoSwiss %s fetch failed: %s", var, exc)

    if not readings:
        return None

    now = utc_now()
    return {
        "source":        f"MeteoSwiss open data (station {TARGET_STATION})",
        "obs_time_utc":  obs_time or now.isoformat(),
        "hour_key":      hour_key(now),
        "wind_speed_kn": _to_knots_from_kmh(readings.get("wind_speed_kmh")),
        "wind_gust_kn":  _to_knots_from_kmh(readings.get("wind_gust_kmh")),
        "wind_dir_deg":  readings.get("wind_dir"),
        "temp_c":        readings.get("temp_c"),
        "precip_mm":     readings.get("precip_mm"),
    }


def _to_knots_from_kmh(kmh) -> float | None:
    v = _float(kmh)
    return round(v / 1.852, 2) if v is not None else None


def _float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def save_measurement(obs: dict) -> None:
    now = utc_now()
    fp  = measurement_file(now)
    fp.write_text(json.dumps(obs, indent=2))
    log.info("  Saved measurement → %s", fp)


# ── 3. Collection entry-point ─────────────────────────────────────────────────

def collect() -> None:
    """One collection cycle: fetch forecasts + measurement and save them."""
    now = utc_now().replace(minute=0, second=0, microsecond=0)

    # Forecasts
    raw = fetch_windguru_forecasts()
    if raw:
        records = parse_windguru_forecasts(raw)
        if records:
            save_forecasts(records, now)

    # Measurement
    obs = fetch_meteobase_measurement()
    if obs:
        save_measurement(obs)


# ── 4. Analysis ───────────────────────────────────────────────────────────────

def load_all_forecasts() -> pd.DataFrame:
    rows = []
    for fp in sorted(FORECASTS_DIR.glob("*.json")):
        payload = json.loads(fp.read_text())
        rows.extend(payload["records"])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["valid_dt"] = pd.to_datetime(df["valid_time_utc"], utc=True)
    df["init_dt"]  = pd.to_datetime(df["init_time_utc"],  utc=True)
    return df


def load_all_measurements() -> pd.DataFrame:
    rows = []
    for fp in sorted(MEASUREMENTS_DIR.glob("*.json")):
        rows.append(json.loads(fp.read_text()))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["obs_dt"] = pd.to_datetime(df["obs_time_utc"], utc=True)
    return df


def _mae(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if mask.sum() == 0:
        return float("nan")
    return (a[mask] - b[mask]).abs().mean()


def _rmse(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if mask.sum() == 0:
        return float("nan")
    return math.sqrt(((a[mask] - b[mask]) ** 2).mean())


def _wind_dir_error(a: pd.Series, b: pd.Series) -> float:
    """Mean absolute circular error for wind direction."""
    mask = a.notna() & b.notna()
    if mask.sum() == 0:
        return float("nan")
    diff = (a[mask] - b[mask]).abs() % 360
    diff = diff.apply(lambda x: 360 - x if x > 180 else x)
    return diff.mean()


def analyse() -> None:
    log.info("═" * 60)
    log.info("ANALYSIS")
    log.info("═" * 60)

    fcst_df = load_all_forecasts()
    obs_df  = load_all_measurements()

    if fcst_df.empty:
        print("\n⚠  No forecast data found in weather_data/forecasts/.\n"
              "   Run the collector first (hourly cron or: python weather_tracker.py --collect).\n")
        return

    if obs_df.empty:
        print("\n⚠  No measurement data found in weather_data/measurements/.\n"
              "   Run the collector first.\n")
        return

    # Round obs to nearest hour for joining
    obs_df["hour_key"] = obs_df["obs_dt"].dt.strftime("%Y-%m-%dT%H")

    # Keep only the forecast rows whose valid_time falls on an hour for which
    # we have an actual observation
    obs_hours = set(obs_df["hour_key"])
    fcst_df   = fcst_df[fcst_df["hour_key"].isin(obs_hours)].copy()

    if fcst_df.empty:
        print("\n⚠  Forecast valid times do not overlap with measurement times yet.\n"
              "   Keep collecting – you need at least one future hour to elapse.\n")
        return

    # Merge
    obs_slim = obs_df[[
        "hour_key", "wind_speed_kn", "wind_gust_kn", "wind_dir_deg", "temp_c"
    ]].rename(columns={
        "wind_speed_kn": "obs_wind_speed_kn",
        "wind_gust_kn":  "obs_wind_gust_kn",
        "wind_dir_deg":  "obs_wind_dir_deg",
        "temp_c":        "obs_temp_c",
    })
    merged = fcst_df.merge(obs_slim, on="hour_key", how="inner")

    if merged.empty:
        print("\n⚠  No matching hour_key found between forecasts and observations.\n")
        return

    # ── per-model, per-lead-bucket metrics ───────────────────────────────────
    lead_bins   = [0, 6, 12, 24, 48, 72, 120]
    lead_labels = ["0-6h", "6-12h", "12-24h", "24-48h", "48-72h", "72-120h"]
    merged["lead_bin"] = pd.cut(merged["lead_hours"], bins=lead_bins,
                                labels=lead_labels, right=True)

    variables = {
        "wind_speed": ("wind_speed_kn", "obs_wind_speed_kn", "MAE (kn)"),
        "wind_gust":  ("wind_gust_kn",  "obs_wind_gust_kn",  "MAE (kn)"),
        "wind_dir":   ("wind_dir_deg",   "obs_wind_dir_deg",  "Circ-MAE (°)"),
        "temp":       ("temp_c",         "obs_temp_c",        "MAE (°C)"),
    }

    summary_rows = []
    detail_rows  = []

    for model, grp in merged.groupby("model"):
        n_total = len(grp)
        for var_name, (fcol, ocol, unit) in variables.items():
            if fcol not in grp.columns or ocol not in grp.columns:
                continue
            if var_name == "wind_dir":
                mae_all = _wind_dir_error(grp[fcol], grp[ocol])
            else:
                mae_all = _mae(grp[fcol], grp[ocol])
            rmse_all = _rmse(grp[fcol], grp[ocol])

            summary_rows.append({
                "model":      model,
                "variable":   var_name,
                "unit":       unit,
                "n_hours":    n_total,
                "MAE_all":    round(mae_all, 3),
                "RMSE_all":   round(rmse_all, 3),
            })

            for lb in lead_labels:
                sub = grp[grp["lead_bin"] == lb]
                if len(sub) == 0:
                    continue
                if var_name == "wind_dir":
                    mae_lb = _wind_dir_error(sub[fcol], sub[ocol])
                else:
                    mae_lb = _mae(sub[fcol], sub[ocol])
                detail_rows.append({
                    "model":     model,
                    "variable":  var_name,
                    "lead_bin":  lb,
                    "n_hours":   len(sub),
                    "MAE":       round(mae_lb, 3),
                })

    summary_df = pd.DataFrame(summary_rows)
    detail_df  = pd.DataFrame(detail_rows)

    # ── composite score (rank) ────────────────────────────────────────────────
    # Weight: wind_speed 40%, wind_gust 30%, wind_dir 15%, temp 15%
    weights = {"wind_speed": 0.40, "wind_gust": 0.30, "wind_dir": 0.15, "temp": 0.15}

    # Normalise each variable's MAE across models (0=best, 1=worst)
    score_df = summary_df.pivot(index="model", columns="variable", values="MAE_all")
    for col in score_df.columns:
        col_min = score_df[col].min()
        col_max = score_df[col].max()
        rng = col_max - col_min
        score_df[col] = (score_df[col] - col_min) / rng if rng else 0.0

    score_df["composite_score"] = sum(
        score_df.get(v, 0) * w for v, w in weights.items()
        if v in score_df.columns
    )
    score_df = score_df.sort_values("composite_score")

    # ── print results ─────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  WINDGURU MODEL ACCURACY REPORT")
    print(f"  Spot: Bielersee, Nidau  |  Observation source: WSA-Ipsach")
    print(f"  Period: {obs_df['hour_key'].min()} → {obs_df['hour_key'].max()}")
    print(f"  Observations used: {len(obs_df)}  |  Forecast records matched: {len(merged)}")
    print("═" * 70)

    print("\n── OVERALL MAE (all lead times) ──────────────────────────────────────")
    pivot = summary_df.pivot_table(
        index="model", columns="variable", values="MAE_all", aggfunc="first"
    ).round(3)
    print(tabulate(pivot, headers="keys", tablefmt="rounded_outline", floatfmt=".3f"))

    print("\n── COMPOSITE RANKING (lower = more accurate) ─────────────────────────")
    rank_table = score_df.reset_index()[["model", "composite_score"]].copy()
    rank_table.insert(0, "rank", range(1, len(rank_table) + 1))
    print(tabulate(rank_table, headers=rank_table.columns,
                   tablefmt="rounded_outline", floatfmt=".4f", index=False))

    best_model = score_df.index[0]
    print(f"\n  ★  BEST MODEL OVERALL: {best_model}")

    print("\n── WIND SPEED MAE BY LEAD-TIME BUCKET (knots) ────────────────────────")
    ws_detail = detail_df[detail_df["variable"] == "wind_speed"].copy()
    if not ws_detail.empty:
        ws_pivot = ws_detail.pivot_table(
            index="model", columns="lead_bin", values="MAE", aggfunc="first"
        ).round(3)
        ws_pivot = ws_pivot.reindex(columns=[l for l in lead_labels if l in ws_pivot.columns])
        print(tabulate(ws_pivot, headers="keys", tablefmt="rounded_outline", floatfmt=".3f"))

    # ── save to CSV ───────────────────────────────────────────────────────────
    ts = utc_now().strftime("%Y%m%dT%H%M")
    summary_fp = ANALYSIS_DIR / f"summary_{ts}.csv"
    detail_fp  = ANALYSIS_DIR / f"detail_{ts}.csv"
    rank_fp    = ANALYSIS_DIR / f"ranking_{ts}.csv"

    summary_df.to_csv(summary_fp, index=False)
    detail_df.to_csv(detail_fp,   index=False)
    score_df.to_csv(rank_fp)

    print(f"\n  Reports saved to:")
    print(f"    {summary_fp}")
    print(f"    {detail_fp}")
    print(f"    {rank_fp}")
    print("═" * 70 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Windguru forecast tracker & accuracy analyser"
    )
    parser.add_argument(
        "--collect", action="store_true",
        help="Fetch forecasts + measurement now and save (default action)"
    )
    parser.add_argument(
        "--analyse", action="store_true",
        help="Analyse saved data and print accuracy report"
    )
    parser.add_argument(
        "--both", action="store_true",
        help="Collect then analyse"
    )
    args = parser.parse_args()

    if args.analyse:
        analyse()
    elif args.both:
        collect()
        analyse()
    else:
        # Default: collect
        collect()


if __name__ == "__main__":
    main()
