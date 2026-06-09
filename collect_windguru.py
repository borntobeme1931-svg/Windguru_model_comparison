#!/usr/bin/env python3
"""
Windguru forecast collector
============================
Fetches forecasts (up to 5 days ahead) from all Windguru models
for spot 56996 (Bielersee, Nidau) and appends them to a monthly JSONL file.

Run every 3 hours via GitHub Actions (see .github/workflows/collect.yml).

Dependencies:
    pip install requests playwright pandas tabulate
    playwright install chromium
"""

import json
import logging
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# ── configuration ─────────────────────────────────────────────────────────────

WINDGURU_SPOT_ID = 56996

DATA_DIR      = Path("weather_data")
FORECASTS_DIR = DATA_DIR / "forecasts"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"

for d in [FORECASTS_DIR, SNAPSHOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(DATA_DIR / "windguru.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def hour_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H")

def month_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m")

def forecast_file(dt: datetime) -> Path:
    return FORECASTS_DIR / f"{month_key(dt)}.jsonl"

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

# Full 16-point compass (German)
_COMPASS_16 = [
    (  0.0, "N"),   ( 22.5, "NNO"), ( 45.0, "NO"),  ( 67.5, "ONO"),
    ( 90.0, "O"),   (112.5, "OSO"), (135.0, "SO"),   (157.5, "SSO"),
    (180.0, "S"),   (202.5, "SSW"), (225.0, "SW"),   (247.5, "WSW"),
    (270.0, "W"),   (292.5, "WNW"), (315.0, "NW"),   (337.5, "NNW"),
]

def _deg_to_compass(deg: float | None) -> str | None:
    if deg is None:
        return None
    idx = int((deg % 360 + 11.25) / 22.5) % 16
    return _COMPASS_16[idx][1]

# ── Windguru scraping ─────────────────────────────────────────────────────────

def _windguru_playwright(spot_id: int) -> dict | None:
    """
    Windguru fires one iapi.php?q=forecast request PER MODEL as the page loads.
    We intercept ALL of them, then stitch them into a single multi-model dict
    shaped like: { "fcst": { "1": model_dict, "2": model_dict, ... } }
    """
    if not PLAYWRIGHT_AVAILABLE:
        log.error("Playwright not installed.")
        return None

    all_responses: list[dict] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx     = browser.new_context()
        page    = ctx.new_page()

        def handle_response(response):
            if "iapi.php" in response.url and "forecast" in response.url:
                try:
                    all_responses.append(response.json())
                except Exception:
                    pass

        page.on("response", handle_response)
        page.goto(f"https://www.windguru.cz/{spot_id}", timeout=30_000)
        # Wait long enough for all per-model XHRs to complete (~15s stagger)
        page.wait_for_timeout(20_000)
        browser.close()

    if not all_responses:
        return None

    log.info("  ✓ Got forecast via Playwright (%d model response(s))", len(all_responses))

    # Stitch into shape B: { "fcst": { "1": {...}, "2": {...}, ... } }
    merged_fcst: dict[str, dict] = {}
    for i, resp in enumerate(all_responses):
        if "hours" in resp or "WINDSPD" in resp:
            model_dict = resp
        elif isinstance(resp.get("fcst"), dict) and (
            "hours" in resp["fcst"] or "WINDSPD" in resp["fcst"]
        ):
            model_dict = {**resp, **resp["fcst"]}
        else:
            model_dict = resp
        merged_fcst[str(i + 1)] = model_dict

    return {"fcst": merged_fcst}


def fetch_windguru_forecasts(spot_id: int = WINDGURU_SPOT_ID) -> dict | None:
    log.info("Fetching Windguru forecasts for spot %s …", spot_id)
    data = _windguru_playwright(spot_id)
    if data and "fcst" in data:
        log.info("  ✓ Got %d model(s) via Playwright", len(data["fcst"]))
        return data
    log.error("  ✗ Could not retrieve Windguru forecasts.")
    return None


def _normalise_to_model_blocks(raw: dict) -> list[dict]:
    fcst_raw = raw.get("fcst", {})

    if isinstance(fcst_raw, dict):
        children = [v for v in fcst_raw.values() if isinstance(v, dict)]
        if children and any("hours" in c or "WINDSPD" in c for c in children):
            return children

    if isinstance(fcst_raw, list):
        children = [v for v in fcst_raw if isinstance(v, dict)]
        if children and any("hours" in c or "WINDSPD" in c for c in children):
            return children

    if "hours" in raw or "WINDSPD" in raw or "fcst_hour_idx" in raw:
        md = dict(raw)
        if isinstance(fcst_raw, dict):
            md.update(fcst_raw)
        return [md]

    if isinstance(fcst_raw, dict) and ("hours" in fcst_raw or "WINDSPD" in fcst_raw):
        return [{**raw, **fcst_raw}]

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

        hours_offsets = md.get("hours") or []
        epoch_times   = md.get("fcst_hour_idx") or []

        wind_speed = md.get("WINDSPD", [])
        wind_gust  = md.get("GUST",    [])
        wind_dir   = md.get("WINDDIR", [])
        temp       = md.get("TMP",     [])
        precip     = md.get("APCP",    [])

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
                "wind_speed_kn":  _safe(wind_speed, i),
                "wind_gust_kn":   _safe(wind_gust,  i),
                "wind_dir_deg":   _safe(wind_dir,   i),
                "wind_dir_txt":   _deg_to_compass(_safe(wind_dir, i)),
                "temp_c":         _safe(temp,        i),
                "precip_mm":      _safe(precip,      i),
            })

    log.info("  Parsed %d hourly forecast records across %d model block(s).",
             len(records), len(model_blocks))
    if not records and model_blocks:
        log.warning("  Sample model dump: %s",
                    {k: str(v)[:80] for k, v in model_blocks[0].items()})
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


def save_snapshot(records: list[dict], run_time: datetime) -> None:
    """
    Write a CSV with one row per model for the current hour.
    Columns: source, wind_speed_kn, wind_gust_kn, wind_dir_deg, wind_dir_txt
    A separate WSA row is added by collect_wsa.py when it runs.
    File: weather_data/snapshots/YYYY-MM-DDTHH.csv  (overwritten each run)
    """
    import csv

    fp = SNAPSHOTS_DIR / f"{run_time.strftime('%Y-%m-%dT%H')}.csv"
    current_hk = hour_key(run_time)

    by_model: dict[str, list[dict]] = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r)

    rows = []
    for model, model_records in sorted(by_model.items()):
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

    with fp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["source", "wind_speed_kn", "wind_gust_kn",
                           "wind_dir_deg", "wind_dir_txt"])
        writer.writeheader()
        writer.writerows(rows)

    log.info("  Snapshot saved → %s  (%d model(s))", fp, len(by_model))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    now = utc_now().replace(minute=0, second=0, microsecond=0)
    raw = fetch_windguru_forecasts()
    if raw:
        records = parse_windguru_forecasts(raw)
        if records:
            save_forecasts(records, now)
            save_snapshot(records, now)
        else:
            log.error("  ✗ Parser returned 0 records.")
    else:
        log.error("  ✗ No data retrieved.")


if __name__ == "__main__":
    main()
