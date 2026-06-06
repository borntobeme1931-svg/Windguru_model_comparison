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
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import pandas as pd
from tabulate import tabulate

# ── optional browser import ──────────────────────────────────────────────────
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# ── configuration ────────────────────────────────────────────────────────────

WINDGURU_SPOT_ID = 56996
STATION_LAT      = 47.113
STATION_LON      = 7.224

DATA_DIR          = Path("weather_data")
FORECASTS_DIR     = DATA_DIR / "forecasts"
MEASUREMENTS_DIR  = DATA_DIR / "measurements"
ANALYSIS_DIR      = DATA_DIR / "analysis"

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
    """YYYY-MM — used to name monthly files."""
    return dt.strftime("%Y-%m")

def forecast_file(run_time: datetime) -> Path:
    return FORECASTS_DIR / f"{month_key(run_time)}.jsonl"

def measurement_file(obs_time: datetime) -> Path:
    return MEASUREMENTS_DIR / f"{month_key(obs_time)}.jsonl"

def _safe(lst, i):
    try:
        v = lst[i]
        return float(v) if v is not None else None
    except (IndexError, TypeError, ValueError):
        return None

def _float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def _to_knots_from_kmh(kmh) -> float | None:
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


def parse_windguru_forecasts(raw: dict) -> list[dict]:
    records = []
    fcst_raw = raw.get("fcst", {})

    if isinstance(fcst_raw, list):
        fcst_block = {str(i): m for i, m in enumerate(fcst_raw) if isinstance(m, dict)}
    elif isinstance(fcst_raw, dict):
        fcst_block = {k: v for k, v in fcst_raw.items() if isinstance(v, dict)}
    else:
        fcst_block = {}

    for model_id, model_data in fcst_block.items():
        model_name = model_data.get("model_name") or model_data.get("name") or model_id
        timestamps = model_data.get("fcst_hour_idx", [])

        wind_speed = model_data.get("WINDSPD", [])
        wind_gust  = model_data.get("GUST",    [])
        wind_dir   = model_data.get("WINDDIR", [])
        temp       = model_data.get("TMP",     [])
        precip     = model_data.get("APCP",    [])

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
                continue

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
    if not records:
        log.warning("  No records parsed. Top-level keys: %s", list(raw.keys()))
        log.warning("  fcst type: %s, sample: %s",
                    type(raw.get("fcst")).__name__, str(raw.get("fcst"))[:300])
    return records


def save_forecasts(records: list[dict], run_time: datetime) -> None:
    """Append forecast records to the monthly JSONL file (one JSON object per line)."""
    fp = forecast_file(run_time)
    entry = {
        "fetch_time_utc": run_time.isoformat(),
        "spot_id":        WINDGURU_SPOT_ID,
        "records":        records,
    }
    with fp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    log.info("  Appended %d forecast records → %s", len(records), fp)


# ── 2. WSA-Ipsach measurements ───────────────────────────────────────────────

def fetch_meteobase_measurement() -> dict | None:
    log.info("Fetching WSA-Ipsach measurement …")

    # ── attempt 1: direct JSON endpoint ─────────────────────────────────────
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
                    log.info("  ✓ Got measurement via JSON endpoint: %s", url)
                    return parsed
        except Exception as exc:
            log.debug("  JSON endpoint failed (%s): %s", url, exc)

    # ── attempt 2: Playwright headless scrape ────────────────────────────────
    parsed = _scrape_wsa_playwright()
    if parsed:
        return parsed

    log.error("  ✗ Could not retrieve any measurement.")
    return None


def _parse_meteobase_json(data: dict) -> dict | None:
    try:
        now = utc_now()
        return {
            "source":        "wsa-ipsach.meteobase.ch (json)",
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


def _scrape_wsa_playwright() -> dict | None:
    """
    Use a headless browser to load WSA-Ipsach and extract sensor readings
    directly from the rendered DOM.  Logs the raw readings on the first run
    so you can verify the values look correct.
    """
    if not PLAYWRIGHT_AVAILABLE:
        log.warning("  Playwright not available — skipping browser scrape.")
        return None

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page    = browser.new_page()
            page.goto("https://wsa-ipsach.meteobase.ch/", timeout=20_000)
            page.wait_for_timeout(3_000)  # let JS finish rendering

            # Extract labelled values from the DOM by scanning table rows.
            # meteoBase.ch renders a table with label | value | unit columns.
            readings = page.evaluate("""() => {
                const result = {};

                // Strategy 1: look for elements with id containing sensor names
                const idMap = {
                    wind_speed: ['windspeed', 'wind_speed', 'ws', 'windmittel', 'geschwindigkeit'],
                    wind_gust:  ['windgust', 'wind_gust', 'wg', 'boe', 'boen', 'gust'],
                    wind_dir:   ['winddir', 'wind_dir', 'wd', 'richtung', 'direction'],
                    temp:       ['temperature', 'temp', 'ta', 'lufttemperatur'],
                    precip:     ['rain', 'precip', 'regen', 'niederschlag', 'rr'],
                };
                for (const [key, ids] of Object.entries(idMap)) {
                    for (const id of ids) {
                        const el = document.getElementById(id)
                                || document.querySelector(`[id*="${id}"]`)
                                || document.querySelector(`[class*="${id}"]`);
                        if (el && el.innerText.match(/[-\d.,]+/)) {
                            result[key] = el.innerText.match(/[-\d.,]+/)[0];
                            break;
                        }
                    }
                }

                // Strategy 2: scan every table row for label+value pairs
                if (Object.keys(result).length < 2) {
                    document.querySelectorAll('tr').forEach(row => {
                        const cells = Array.from(row.querySelectorAll('td, th'));
                        if (cells.length < 2) return;
                        const label = (cells[0].innerText || '').toLowerCase().trim();
                        const valText = (cells[1].innerText || cells[2]?.innerText || '').trim();
                        const num = valText.match(/^[-\d.,]+/);
                        if (!num) return;
                        const v = num[0].replace(',', '.');

                        if (!result.wind_speed && (label.includes('wind') && !label.includes('bö') && !label.includes('gust') && !label.includes('richtung') && !label.includes('dir')))
                            result.wind_speed = v;
                        if (!result.wind_gust && (label.includes('bö') || label.includes('gust') || label.includes('boe')))
                            result.wind_gust = v;
                        if (!result.wind_dir && (label.includes('richtung') || label.includes('direction') || label.includes('dir')))
                            result.wind_dir = v;
                        if (!result.temp && (label.includes('temp') || label.includes('°c')))
                            result.temp = v;
                        if (!result.precip && (label.includes('regen') || label.includes('rain') || label.includes('niederschlag') || label.includes('precip')))
                            result.precip = v;
                    });
                }

                return result;
            }""")

            # Also grab visible text for debugging if readings are sparse
            if len(readings) < 2:
                body_text = page.inner_text("body")
                log.warning("  WSA-Ipsach: only %d fields extracted. Page text snippet:\n%s",
                            len(readings), body_text[:1500])
            else:
                log.info("  WSA-Ipsach DOM readings: %s", readings)

            browser.close()

        now = utc_now()
        wind_kmh = _float(readings.get("wind_speed"))
        gust_kmh = _float(readings.get("wind_gust"))

        if wind_kmh is None and _float(readings.get("temp")) is None:
            log.warning("  WSA-Ipsach Playwright: page loaded but no usable values found.")
            return None

        result = {
            "source":        "wsa-ipsach.meteobase.ch (playwright)",
            "obs_time_utc":  now.isoformat(),
            "hour_key":      hour_key(now),
            "wind_speed_kn": _to_knots_from_kmh(wind_kmh),
            "wind_gust_kn":  _to_knots_from_kmh(gust_kmh),
            "wind_dir_deg":  _float(readings.get("wind_dir")),
            "temp_c":        _float(readings.get("temp")),
            "precip_mm":     _float(readings.get("precip")),
        }
        log.info("  ✓ Got WSA-Ipsach measurement via Playwright: %s", result)
        return result

    except Exception as exc:
        log.warning("  WSA-Ipsach Playwright scrape failed: %s", exc)
        return None



def save_measurement(obs: dict) -> None:
    """Append measurement to the monthly JSONL file. Skips if this hour already recorded."""
    now = utc_now()
    fp  = measurement_file(now)
    hk  = hour_key(now)

    # Avoid duplicate entries: skip if this hour is already in the file
    if fp.exists():
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    if json.loads(line).get("hour_key") == hk:
                        log.info("  Measurement for %s already recorded — skipping.", hk)
                        return
                except Exception:
                    pass

    with fp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obs) + "\n")
    log.info("  Appended measurement → %s", fp)


# ── 3. Collection entry-point ─────────────────────────────────────────────────

def collect() -> None:
    now = utc_now().replace(minute=0, second=0, microsecond=0)

    raw = fetch_windguru_forecasts()
    if raw:
        records = parse_windguru_forecasts(raw)
        if records:
            save_forecasts(records, now)

    obs = fetch_meteobase_measurement()
    if obs:
        save_measurement(obs)


# ── 4. Analysis ───────────────────────────────────────────────────────────────

def load_all_forecasts() -> pd.DataFrame:
    """Load all forecast records from monthly JSONL files."""
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
    """Load all measurements from monthly JSONL files."""
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
            mae_all  = _wind_dir_error(grp[fcol], grp[ocol]) if var_name == "wind_dir" else _mae(grp[fcol], grp[ocol])
            rmse_all = _rmse(grp[fcol], grp[ocol])
            summary_rows.append({
                "model": model, "variable": var_name, "unit": unit,
                "n_hours": n_total, "MAE_all": round(mae_all, 3), "RMSE_all": round(rmse_all, 3),
            })
            for lb in lead_labels:
                sub = grp[grp["lead_bin"] == lb]
                if len(sub) == 0:
                    continue
                mae_lb = _wind_dir_error(sub[fcol], sub[ocol]) if var_name == "wind_dir" else _mae(sub[fcol], sub[ocol])
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
        score_df.get(v, 0) * w for v, w in weights.items() if v in score_df.columns
    )
    score_df = score_df.sort_values("composite_score")

    print("\n" + "═" * 70)
    print("  WINDGURU MODEL ACCURACY REPORT")
    print(f"  Spot: Bielersee, Nidau  |  Observation source: WSA-Ipsach")
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
        ws_pivot = ws_pivot.reindex(columns=[l for l in lead_labels if l in ws_pivot.columns])
        print(tabulate(ws_pivot, headers="keys", tablefmt="rounded_outline", floatfmt=".3f"))

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
    parser = argparse.ArgumentParser(description="Windguru forecast tracker & accuracy analyser")
    parser.add_argument("--collect", action="store_true", help="Fetch and save data (default)")
    parser.add_argument("--analyse", action="store_true", help="Print accuracy report")
    parser.add_argument("--both",    action="store_true", help="Collect then analyse")
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
