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

Dependencies:
    pip install requests playwright pandas tabulate
    playwright install chromium
"""

import argparse
import json
import logging
import math
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import pandas as pd
from tabulate import tabulate

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# ── configuration ────────────────────────────────────────────────────────────

WINDGURU_SPOT_ID = 56996
STATION_LAT      = 47.113
STATION_LON      = 7.224

DATA_DIR         = Path("weather_data")
FORECASTS_DIR    = DATA_DIR / "forecasts"
MEASUREMENTS_DIR = DATA_DIR / "measurements"
ANALYSIS_DIR     = DATA_DIR / "analysis"

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
    return dt.strftime("%Y-%m-%dT%H")

def month_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m")

def forecast_file(dt: datetime) -> Path:
    return FORECASTS_DIR / f"{month_key(dt)}.jsonl"

def measurement_file(dt: datetime) -> Path:
    return MEASUREMENTS_DIR / f"{month_key(dt)}.jsonl"

def _safe(lst, i):
    try:
        v = lst[i]
        return float(v) if v is not None else None
    except (IndexError, TypeError, ValueError):
        return None

def _float(v) -> float | None:
    try:
        return float(str(v).replace(",", "."))
    except (TypeError, ValueError):
        return None

def _knots_from_ms(ms) -> float | None:
    v = _float(ms)
    return round(v * 1.94384, 2) if v is not None else None

def _knots_from_kmh(kmh) -> float | None:
    v = _float(kmh)
    return round(v / 1.852, 2) if v is not None else None


# ── 1. Windguru forecast scraping ────────────────────────────────────────────

def _windguru_internal_api(spot_id: int) -> dict | None:
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
        uid_match = re.search(r'"uid"\s*:\s*"([^"]+)"', page_resp.text)
        salt = uid_match.group(1) if uid_match else str(int(time.time()))
        import hashlib
        hash_val = hashlib.md5(f"{salt}{spot_id}".encode()).hexdigest()
        params = {
            "q": "forecast", "id_spot": spot_id, "uid": salt,
            "hash": hash_val, "lang": "en", "no_wave": 0, "WGCACHEABLE": 21600,
        }
        api_resp = session.get("https://www.windguru.cz/int/iapi.php",
                               params=params, timeout=20)
        api_resp.raise_for_status()
        return api_resp.json()
    except Exception as exc:
        log.warning("Internal API call failed: %s", exc)
        return None


def _windguru_playwright(spot_id: int) -> dict | None:
    if not PLAYWRIGHT_AVAILABLE:
        log.error("Playwright not installed.")
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
        page.wait_for_timeout(8_000)
        browser.close()

    return captured


def fetch_windguru_forecasts(spot_id: int = WINDGURU_SPOT_ID) -> dict | None:
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


def _normalise_to_model_blocks(raw: dict) -> list[dict]:
    """
    Windguru can return forecast data in several shapes:
      A) Single model flat:  raw itself has 'hours', 'WINDSPD', 'initdate', etc.
         plus top-level 'model' / 'id_model' / 'wgmodel'.
      B) Multi-model dict:   raw['fcst'] = { "1": {model_dict}, "2": {…}, … }
      C) Multi-model list:   raw['fcst'] = [ {model_dict}, … ]

    Returns a list of model dicts, each guaranteed to have the forecast arrays.
    """
    fcst_raw = raw.get("fcst", {})

    # shape B – dict of model dicts
    if isinstance(fcst_raw, dict):
        children = [v for v in fcst_raw.values() if isinstance(v, dict)]
        if children and any("hours" in c or "WINDSPD" in c for c in children):
            return children

    # shape C – list of model dicts
    if isinstance(fcst_raw, list):
        children = [v for v in fcst_raw if isinstance(v, dict)]
        if children and any("hours" in c or "WINDSPD" in c for c in children):
            return children

    # shape A – the forecast arrays live directly in raw (single model response)
    if "hours" in raw or "WINDSPD" in raw or "fcst_hour_idx" in raw:
        # merge top-level model metadata with whatever is in fcst (if dict)
        md = dict(raw)
        if isinstance(fcst_raw, dict):
            md.update(fcst_raw)
        return [md]

    # shape A variant – arrays are inside fcst dict but no child dicts
    if isinstance(fcst_raw, dict) and ("hours" in fcst_raw or "WINDSPD" in fcst_raw):
        md = {**raw, **fcst_raw}
        return [md]

    return []


def parse_windguru_forecasts(raw: dict) -> list[dict]:
    records = []
    model_blocks = _normalise_to_model_blocks(raw)

    if not model_blocks:
        log.warning("  Could not find forecast arrays. Raw keys: %s", list(raw.keys()))
        log.warning("  fcst sample: %s", str(raw.get("fcst"))[:300])
        return []

    log.info("  Found %d model block(s). First block keys: %s",
             len(model_blocks), list(model_blocks[0].keys())[:25])

    for md in model_blocks:
        model_name = (md.get("model_name") or md.get("name") or
                      str(md.get("id_model") or md.get("wgmodel") or "unknown"))

        # ── resolve init datetime ────────────────────────────────────────────
        # Windguru stores it as "YYYY-MM-DD HH:MM" in initdate, or epoch in init_d
        init_dt = None
        raw_init = md.get("initdate") or md.get("init_date")
        if raw_init:
            for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
                try:
                    init_dt = datetime.strptime(raw_init, fmt).replace(tzinfo=timezone.utc)
                    break
                except ValueError:
                    pass
        if init_dt is None:
            raw_ts = md.get("init_d") or md.get("init_timestamp")
            try:
                init_dt = datetime.fromtimestamp(float(raw_ts), tz=timezone.utc)
            except (TypeError, ValueError):
                init_dt = utc_now().replace(minute=0, second=0, microsecond=0)

        # ── resolve time axis ────────────────────────────────────────────────
        # "hours" = list of integer offsets from init_dt (most common)
        # "fcst_hour_idx" = list of epoch timestamps (alternative)
        hours_offsets = md.get("hours") or []          # [0, 1, 2, 3, ...]
        epoch_times   = md.get("fcst_hour_idx") or []  # [1234567890, ...]

        wind_speed = md.get("WINDSPD", [])
        wind_gust  = md.get("GUST",    [])
        wind_dir   = md.get("WINDDIR", [])
        temp       = md.get("TMP",     [])
        precip     = md.get("APCP",    [])

        # use whichever time axis is populated
        if hours_offsets:
            time_iter = [(i, init_dt + timedelta(hours=int(h)))
                         for i, h in enumerate(hours_offsets)]
        elif epoch_times:
            time_iter = [(i, datetime.fromtimestamp(float(ts), tz=timezone.utc))
                         for i, ts in enumerate(epoch_times)]
        else:
            log.warning("  Model %s: no time axis found, skipping.", model_name)
            continue

        for i, valid_dt in time_iter:
            lead = int((valid_dt - init_dt).total_seconds() / 3600)
            if lead < 0 or lead > 5 * 24:
                continue

            records.append({
                "model":          str(model_name),
                "init_time_utc":  init_dt.isoformat(),
                "valid_time_utc": valid_dt.isoformat(),
                "hour_key":       hour_key(valid_dt),
                "lead_hours":     lead,
                "wind_speed_kn":  _safe(wind_speed, i),  # already knots from Windguru
                "wind_gust_kn":   _safe(wind_gust,  i),
                "wind_dir_deg":   _safe(wind_dir,   i),
                "wind_dir_txt":   _deg_to_compass(_safe(wind_dir, i)),
                "temp_c":         _safe(temp,        i),
                "precip_mm":      _safe(precip,      i),
            })

    log.info("  Parsed %d hourly forecast records across %d model block(s).",
             len(records), len(model_blocks))
    if not records and model_blocks:
        sample = model_blocks[0]
        log.warning("  Sample model dump: %s",
                    {k: str(v)[:80] for k, v in sample.items()})
    return records


def save_forecasts(records: list[dict], run_time: datetime) -> None:
    fp = forecast_file(run_time)
    entry = {
        "fetch_time_utc": run_time.isoformat(),
        "spot_id":        WINDGURU_SPOT_ID,
        "records":        records,
    }
    with fp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    log.info("  Appended %d forecast records → %s", len(records), fp)



def save_snapshot(records: list[dict], obs: dict | None, run_time: datetime) -> None:
    """
    Write a simple CSV with one row per model showing the current-hour forecast,
    plus one row for the WSA-Ipsach measurement.
    Columns: source, wind_speed_kn, wind_gust_kn, wind_dir_deg, wind_dir_txt
    File:    weather_data/snapshots/YYYY-MM-DDTHH.csv  (one per hour, overwritten)
    """
    snap_dir = DATA_DIR / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    fp = snap_dir / f"{run_time.strftime('%Y-%m-%dT%H')}.csv"

    current_hk = hour_key(run_time)
    rows = []

    # One row per model: use the forecast valid for the current hour,
    # falling back to the nearest available hour for that model.
    by_model: dict[str, list[dict]] = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r)

    for model, model_records in sorted(by_model.items()):
        # prefer exact hour match, else closest future hour
        exact = [r for r in model_records if r["hour_key"] == current_hk]
        if exact:
            r = exact[0]
        else:
            future = sorted(
                [r for r in model_records if r["hour_key"] >= current_hk],
                key=lambda x: x["hour_key"],
            )
            r = future[0] if future else model_records[0]
        rows.append({
            "source":        model,
            "wind_speed_kn": r.get("wind_speed_kn"),
            "wind_gust_kn":  r.get("wind_gust_kn"),
            "wind_dir_deg":  r.get("wind_dir_deg"),
            "wind_dir_txt":  r.get("wind_dir_txt"),
        })

    # WSA measurement row
    if obs:
        rows.append({
            "source":        "WSA-Ipsach (measured)",
            "wind_speed_kn": obs.get("wind_speed_kn"),
            "wind_gust_kn":  obs.get("wind_gust_kn"),
            "wind_dir_deg":  obs.get("wind_dir_deg"),
            "wind_dir_txt":  obs.get("wind_dir_txt"),
        })

    with fp.open("w", encoding="utf-8", newline="") as f:
        import csv
        writer = csv.DictWriter(
            f, fieldnames=["source","wind_speed_kn","wind_gust_kn","wind_dir_deg","wind_dir_txt"]
        )
        writer.writeheader()
        writer.writerows(rows)

    log.info("  Snapshot saved -> %s  (%d model(s)%s)", fp, len(by_model),
             " + WSA" if obs else "")


# ── 2. WSA-Ipsach measurements ───────────────────────────────────────────────

# Full 16-point compass (German: N=Nord, O=Ost, S=Süd, W=West)
_COMPASS_16 = [
    (  0.0, "N"),   ( 22.5, "NNO"), ( 45.0, "NO"),  ( 67.5, "ONO"),
    ( 90.0, "O"),   (112.5, "OSO"), (135.0, "SO"),   (157.5, "SSO"),
    (180.0, "S"),   (202.5, "SSW"), (225.0, "SW"),   (247.5, "WSW"),
    (270.0, "W"),   (292.5, "WNW"), (315.0, "NW"),   (337.5, "NNW"),
]

# Build lookup: German label → degrees  (also accept English O→E variants)
_DIR_TO_DEG: dict[str, float] = {}
for _deg, _lbl in _COMPASS_16:
    _DIR_TO_DEG[_lbl.lower()] = _deg
    _DIR_TO_DEG[_lbl.lower().replace("o", "e")] = _deg   # ONO→ENE etc.


def _dir_to_deg(text: str | None) -> float | None:
    if not text:
        return None
    t = text.strip().lower().replace("-", "").replace(" ", "")
    try:
        v = float(t)
        return v if 0 <= v <= 360 else None
    except ValueError:
        pass
    return _DIR_TO_DEG.get(t)


def _deg_to_compass(deg: float | None) -> str | None:
    if deg is None:
        return None
    idx = int((deg % 360 + 11.25) / 22.5) % 16
    return _COMPASS_16[idx][1]


def fetch_meteobase_measurement() -> dict | None:
    log.info("Fetching WSA-Ipsach measurement ...")
    parsed = _scrape_wsa()
    if parsed:
        return parsed
    log.error("  ✗ Could not retrieve any measurement.")
    return None


def _scrape_wsa() -> dict | None:
    """
    Load wsa-ipsach.meteobase.ch with a headless browser and extract
    wind speed, gust, and direction from the rendered page text.

    Known page format (from live observation):
        Wind-10min-Ø: 4km/h (2.2kn, 1Bf) SW
        Wind-10min-Max: 10.1km/h (5.5kn, 1Bf) W

    Wind speed and gust are in km/h; converted to knots.
    Direction is a German 16-point abbreviation (N, NNO, NO, ONO, O, ... SSW, SW, WSW, W, ...).
    The site shows Wassertemperatur (water temp) — not recorded as air temperature.
    """
    if not PLAYWRIGHT_AVAILABLE:
        log.warning("  Playwright not available.")
        return None

    page_text = ""
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox"],
            )
            page = browser.new_page(
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                )
            )
            # "load" avoids the 30s networkidle timeout
            try:
                page.goto("https://wsa-ipsach.meteobase.ch/",
                          wait_until="load", timeout=20_000)
            except Exception as nav_exc:
                log.warning("  WSA goto warning: %s", nav_exc)

            # Extra wait for deferred JS rendering
            page.wait_for_timeout(3_000)

            try:
                page_text = page.inner_text("body").strip()
            except Exception:
                page_text = page.content()

            browser.close()

    except Exception as exc:
        log.warning("  WSA Playwright error: %s", exc)
        return None

    if not page_text:
        log.warning("  WSA: empty page text.")
        return None

    log.info("  WSA page (%d chars):\n%s", len(page_text), page_text)

    # Pattern: "Wind-10min-Ø: 4km/h (2.2kn, 1Bf) SW"
    # Captures: speed, unit, optional (knots block), optional direction label
    # Format: "Wind-10min-Ø: 4km/h (2.2kn, 1Bf) SW"
    # Capture the knot value from the parentheses directly — more accurate
    # than converting from km/h ourselves.
    WIND_RE = re.compile(
        r"Wind-10min-[\u00d8O]:\s*"
        r"[\d.,]+\s*(?:km/h|m/s)"           # km/h value (ignored)
        r"\s*\(([\d.,]+)\s*kn[^)]*\)"    # (X.Xkn ...) — capture knots
        r"\s*([A-Z]{1,3})?",                  # optional direction
        re.IGNORECASE,
    )
    GUST_RE = re.compile(
        r"Wind-10min-Max:\s*"
        r"[\d.,]+\s*(?:km/h|m/s)"
        r"\s*\(([\d.,]+)\s*kn[^)]*\)",   # (X.Xkn ...)
        re.IGNORECASE,
    )

    ws_kn = wg_kn = wd_deg = wd_txt = None

    m_ws = WIND_RE.search(page_text)
    if m_ws:
        ws_kn  = _float(m_ws.group(1))        # already in knots
        wd_raw = m_ws.group(2)                 # e.g. "SW", "NNO", or None
        wd_deg = _dir_to_deg(wd_raw)
        wd_txt = wd_raw.upper() if wd_raw else None

    m_wg = GUST_RE.search(page_text)
    if m_wg:
        wg_kn = _float(m_wg.group(1))         # already in knots

    log.info("  WSA parsed: wind=%s kn  gust=%s kn  dir=%s (%s deg)",
             ws_kn, wg_kn, wd_txt, wd_deg)

    if ws_kn is None:
        log.warning("  WSA: could not extract wind speed from page text.")
        return None

    now = utc_now()
    return {
        "source":        "wsa-ipsach.meteobase.ch",
        "obs_time_utc":  now.isoformat(),
        "hour_key":      hour_key(now),
        "wind_speed_kn": ws_kn,
        "wind_gust_kn":  wg_kn,
        "wind_dir_deg":  wd_deg,
        "wind_dir_txt":  wd_txt,
        "temp_c":        None,   # site only shows water temp, not air temp
        "precip_mm":     None,
    }



# ── 3. Collection entry-point ─────────────────────────────────────────────────

def save_measurement(obs: dict) -> None:
    """Append to monthly JSONL. Skips if this hour is already recorded."""
    now = utc_now()
    fp  = measurement_file(now)
    hk  = hour_key(now)

    if fp.exists():
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    if json.loads(line.strip()).get("hour_key") == hk:
                        log.info("  Measurement for %s already recorded — skipping.", hk)
                        return
                except Exception:
                    pass

    with fp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obs) + "\n")
    log.info("  Appended measurement -> %s", fp)


def collect() -> None:
    now = utc_now().replace(minute=0, second=0, microsecond=0)

    raw = fetch_windguru_forecasts()
    if raw:
        records = parse_windguru_forecasts(raw)
        if records:
            save_forecasts(records, now)
        else:
            log.error("  ✗ Parser returned 0 records — check log for model key dump above.")

    obs = fetch_meteobase_measurement()
    if obs:
        save_measurement(obs)

    if records or obs:
        save_snapshot(records or [], obs, now)


# ── 4. Analysis ───────────────────────────────────────────────────────────────

def load_all_forecasts() -> pd.DataFrame:
    rows = []
    for fp in sorted(FORECASTS_DIR.glob("*.jsonl")):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    rows.extend(payload.get("records", []))
                except Exception:
                    pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["valid_dt"] = pd.to_datetime(df["valid_time_utc"], utc=True)
    df["init_dt"]  = pd.to_datetime(df["init_time_utc"],  utc=True)
    return df


def load_all_measurements() -> pd.DataFrame:
    rows = []
    for fp in sorted(MEASUREMENTS_DIR.glob("*.jsonl")):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
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
        print("\n⚠  No forecast data found. Run the collector first.\n")
        return
    if obs_df.empty:
        print("\n⚠  No measurement data found. Run the collector first.\n")
        return

    obs_df["hour_key"] = obs_df["obs_dt"].dt.strftime("%Y-%m-%dT%H")
    obs_hours = set(obs_df["hour_key"])
    fcst_df   = fcst_df[fcst_df["hour_key"].isin(obs_hours)].copy()

    if fcst_df.empty:
        print("\n⚠  Forecast valid times do not overlap with measurements yet.\n"
              "   Keep collecting — you need at least one future hour to elapse.\n")
        return

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
        print("\n⚠  No matching hour_key between forecasts and observations.\n")
        return

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
            mae_all  = (_wind_dir_error(grp[fcol], grp[ocol])
                        if var_name == "wind_dir" else _mae(grp[fcol], grp[ocol]))
            rmse_all = _rmse(grp[fcol], grp[ocol])
            summary_rows.append({
                "model": model, "variable": var_name, "unit": unit,
                "n_hours": n_total, "MAE_all": round(mae_all, 3),
                "RMSE_all": round(rmse_all, 3),
            })
            for lb in lead_labels:
                sub = grp[grp["lead_bin"] == lb]
                if len(sub) == 0:
                    continue
                mae_lb = (_wind_dir_error(sub[fcol], sub[ocol])
                          if var_name == "wind_dir" else _mae(sub[fcol], sub[ocol]))
                detail_rows.append({
                    "model": model, "variable": var_name,
                    "lead_bin": lb, "n_hours": len(sub), "MAE": round(mae_lb, 3),
                })

    summary_df = pd.DataFrame(summary_rows)
    detail_df  = pd.DataFrame(detail_rows)

    weights  = {"wind_speed": 0.40, "wind_gust": 0.30, "wind_dir": 0.15, "temp": 0.15}
    score_df = summary_df.pivot(index="model", columns="variable", values="MAE_all")
    for col in score_df.columns:
        col_min, col_max = score_df[col].min(), score_df[col].max()
        rng = col_max - col_min
        score_df[col] = (score_df[col] - col_min) / rng if rng else 0.0
    score_df["composite_score"] = sum(
        score_df.get(v, 0) * w for v, w in weights.items()
        if v in score_df.columns
    )
    score_df = score_df.sort_values("composite_score")

    print("\n" + "═" * 70)
    print("  WINDGURU MODEL ACCURACY REPORT")
    print(f"  Spot: Bielersee, Nidau  |  Observations: WSA-Ipsach")
    print(f"  Period: {obs_df['hour_key'].min()} → {obs_df['hour_key'].max()}")
    print(f"  Observations: {len(obs_df)}  |  Matched forecast records: {len(merged)}")
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
    print(f"\n  ★  BEST MODEL OVERALL: {score_df.index[0]}")

    print("\n── WIND SPEED MAE BY LEAD-TIME BUCKET (knots) ────────────────────────")
    ws_detail = detail_df[detail_df["variable"] == "wind_speed"].copy()
    if not ws_detail.empty:
        ws_pivot = ws_detail.pivot_table(
            index="model", columns="lead_bin", values="MAE", aggfunc="first"
        ).round(3)
        ws_pivot = ws_pivot.reindex(
            columns=[l for l in lead_labels if l in ws_pivot.columns])
        print(tabulate(ws_pivot, headers="keys", tablefmt="rounded_outline",
                       floatfmt=".3f"))

    ts = utc_now().strftime("%Y%m%dT%H%M")
    summary_fp = ANALYSIS_DIR / f"summary_{ts}.csv"
    detail_fp  = ANALYSIS_DIR / f"detail_{ts}.csv"
    rank_fp    = ANALYSIS_DIR / f"ranking_{ts}.csv"
    summary_df.to_csv(summary_fp, index=False)
    detail_df.to_csv(detail_fp,   index=False)
    score_df.to_csv(rank_fp)
    print(f"\n  Reports saved to: {ANALYSIS_DIR}/")
    print("═" * 70 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Windguru forecast tracker & accuracy analyser")
    parser.add_argument("--collect", action="store_true",
                        help="Fetch and save data (default)")
    parser.add_argument("--analyse", action="store_true",
                        help="Print accuracy report")
    parser.add_argument("--both",    action="store_true",
                        help="Collect then analyse")
    args = parser.parse_args()

    if args.analyse:
        analyse()
    elif args.both:
        collect()
        analyse()
    else:
        collect()


if __name__ == "__main__":
    main()
