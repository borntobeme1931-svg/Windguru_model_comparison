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


# ── 2. WSA-Ipsach measurements ───────────────────────────────────────────────

def fetch_meteobase_measurement() -> dict | None:
    log.info("Fetching WSA-Ipsach measurement …")

    # attempt 1: direct JSON endpoint
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
                    log.info("  ✓ Got measurement via JSON endpoint")
                    return parsed
        except Exception as exc:
            log.debug("  JSON endpoint failed (%s): %s", url, exc)

    # attempt 2: Playwright headless scrape
    parsed = _scrape_wsa_playwright()
    if parsed:
        return parsed

    log.error("  ✗ Could not retrieve any measurement.")
    return None


def _parse_meteobase_json(data: dict) -> dict | None:
    try:
        now = utc_now()
        ws_kn = (_knots_from_ms(data.get("wind_speed") or data.get("ws"))
                 or _knots_from_kmh(data.get("wind_speed_kmh")))
        wg_kn = (_knots_from_ms(data.get("wind_gust") or data.get("wg"))
                 or _knots_from_kmh(data.get("wind_gust_kmh")))
        wd_deg = _float(data.get("wind_dir") or data.get("wd"))
        result = {
            "source":        "wsa-ipsach.meteobase.ch (json)",
            "obs_time_utc":  now.isoformat(),
            "hour_key":      hour_key(now),
            "wind_speed_kn": ws_kn,
            "wind_gust_kn":  wg_kn,
            "wind_dir_deg":  wd_deg,
            "wind_dir_txt":  _deg_to_compass(wd_deg),
            "temp_c":        _float(data.get("temp") or data.get("temperature") or data.get("ta")),
            "precip_mm":     _float(data.get("precip") or data.get("rain") or data.get("rr")),
        }
        # only return if we got at least wind speed or temp
        if result["wind_speed_kn"] is not None or result["temp_c"] is not None:
            return result
        return None
    except Exception:
        return None


# Full 16-point compass: label → degrees  (German abbreviations used on WSA)
# N=Nord, O=Ost, S=Süd, W=West
_COMPASS_16 = [
    (  0.0, "N"),
    ( 22.5, "NNO"),
    ( 45.0, "NO"),
    ( 67.5, "ONO"),
    ( 90.0, "O"),
    (112.5, "OSO"),
    (135.0, "SO"),
    (157.5, "SSO"),
    (180.0, "S"),
    (202.5, "SSW"),
    (225.0, "SW"),
    (247.5, "WSW"),
    (270.0, "W"),
    (292.5, "WNW"),
    (315.0, "NW"),
    (337.5, "NNW"),
]

# German + English abbreviation → degrees
_DIR_TO_DEG: dict[str, float] = {}
for _deg, _lbl in _COMPASS_16:
    _DIR_TO_DEG[_lbl.lower()] = _deg
    # map O→E variants for English input
    _en = _lbl.replace("O", "E").replace("o", "e")
    _DIR_TO_DEG[_en.lower()] = _deg

def _dir_to_deg(text: str) -> float | None:
    """Convert German/English cardinal direction text to degrees."""
    if not text:
        return None
    cleaned = text.strip().lower().replace("-", "").replace(".", "").replace(" ", "")
    try:
        v = float(cleaned)
        return v if 0 <= v <= 360 else None
    except ValueError:
        pass
    return _DIR_TO_DEG.get(cleaned)

def _deg_to_compass(deg: float | None) -> str | None:
    """Convert degrees to 16-point compass label (German, e.g. WSW)."""
    if deg is None:
        return None
    deg = deg % 360
    # each sector is 22.5° wide; find nearest
    idx = int((deg + 11.25) / 22.5) % 16
    return _COMPASS_16[idx][1]


def _scrape_wsa_playwright() -> dict | None:
    """
    Intercept the XHR/fetch API calls that wsa-ipsach.meteobase.ch makes to its
    data backend — the same network-interception technique used for Windguru.
    This is more robust than waiting for the page to render, and sidesteps any
    anti-bot measures on the HTML response.

    Strategy:
      1. Capture every JSON response from the meteobase.ch domain while the page
         loads.  The backend typically calls something like /api/current or
         /api/data or passes JSON in a query-string response.
      2. Walk all captured payloads looking for wind / temperature keys.
      3. Fall back to parsing the rendered page text if no JSON API is found
         (e.g. the site switched to server-side rendering).
    """
    if not PLAYWRIGHT_AVAILABLE:
        log.warning("  Playwright not available — skipping browser scrape.")
        return None

    captured_responses: list[dict] = []   # all JSON payloads from the domain
    captured_page_text: list[str]  = []   # rendered body text (fallback)

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox"],
            )
            ctx  = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                extra_http_headers={"Accept-Language": "de-CH,de;q=0.9,en;q=0.8"},
            )
            page = ctx.new_page()

            # ── intercept all responses from the meteobase domain ────────────
            def on_response(response):
                try:
                    if "meteobase.ch" not in response.url:
                        return
                    ct = response.headers.get("content-type", "")
                    if "json" in ct or response.url.endswith((".json", ".php")):
                        body = response.json()
                        captured_responses.append({"url": response.url, "body": body})
                        log.debug("  WSA XHR captured: %s → %s",
                                  response.url, str(body)[:200])
                except Exception:
                    pass  # non-JSON or empty body — ignore

            page.on("response", on_response)

            try:
                page.goto("https://wsa-ipsach.meteobase.ch/",
                          wait_until="networkidle", timeout=30_000)
            except Exception as nav_exc:
                log.warning("  WSA navigation warning: %s", nav_exc)

            # Give JS a moment to fire any deferred data fetches
            page.wait_for_timeout(4_000)

            # Grab rendered text as fallback
            try:
                txt = page.inner_text("body").strip()
                if txt:
                    captured_page_text.append(txt)
            except Exception:
                pass

            browser.close()

    except Exception as exc:
        log.warning("  WSA-Ipsach Playwright failed: %s", exc)
        return None

    # ── attempt 1: parse captured JSON API responses ─────────────────────────
    if captured_responses:
        log.info("  WSA: intercepted %d JSON response(s)", len(captured_responses))
        for item in captured_responses:
            parsed = _parse_wsa_api_response(item["url"], item["body"])
            if parsed:
                log.info("  ✓ WSA-Ipsach from XHR: wind=%s kn gust=%s kn "
                         "dir=%s (%s°) temp=%s°C",
                         parsed.get("wind_speed_kn"), parsed.get("wind_gust_kn"),
                         parsed.get("wind_dir_txt"), parsed.get("wind_dir_deg"),
                         parsed.get("temp_c"))
                return parsed
        log.debug("  WSA: JSON responses found but no wind/temp data extracted.")

    # ── attempt 2: parse rendered page text ──────────────────────────────────
    page_text = "\n".join(captured_page_text)
    if page_text:
        log.info("  WSA: falling back to page-text parse (%d chars):\n%s", len(page_text), page_text)
        log.debug("  WSA page text sample:\n%s", page_text[:3000])
        parsed = _parse_wsa_page_text(page_text)
        if parsed:
            log.info("  ✓ WSA-Ipsach from page text: wind=%s kn gust=%s kn "
                     "dir=%s (%s°) temp=%s°C",
                     parsed.get("wind_speed_kn"), parsed.get("wind_gust_kn"),
                     parsed.get("wind_dir_txt"), parsed.get("wind_dir_deg"),
                     parsed.get("temp_c"))
            return parsed

    log.warning("  WSA-Ipsach: no usable data found in XHR or page text.")
    return None


def _parse_wsa_api_response(url: str, data) -> dict | None:
    """
    Try to extract weather readings from a JSON payload captured from the
    meteobase.ch backend.  The payload can be:
      - a dict with sensor keys at the top level
      - a list of sensor dicts  {"name": "WindSpeed", "value": 14.2, "unit": "km/h"}
      - a nested structure  {"data": {"sensors": [...]}}
    """
    now = utc_now()

    def _search(obj, depth=0) -> dict:
        """Recursively search obj for wind/temp keys; return best candidate dict."""
        if depth > 5:
            return {}
        if isinstance(obj, dict):
            keys_lc = {k.lower(): k for k in obj}
            # direct hit: dict has wind speed key
            wind_hits = [k for k in keys_lc if any(
                tok in k for tok in ("wind", "ws", "windspd", "geschw"))]
            if wind_hits:
                return obj
            # recurse into values
            best = {}
            for v in obj.values():
                candidate = _search(v, depth + 1)
                if len(candidate) > len(best):
                    best = candidate
            return best
        if isinstance(obj, list):
            # list of sensor-dicts: [{"name": "...", "value": ..., "unit": "..."}]
            if all(isinstance(x, dict) and "name" in x for x in obj[:5]):
                merged: dict = {}
                for sensor in obj:
                    name = str(sensor.get("name", "")).lower().strip()
                    val  = sensor.get("value") or sensor.get("val")
                    unit = str(sensor.get("unit", "")).lower()
                    merged[name] = {"value": val, "unit": unit}
                return merged
            best = {}
            for item in obj:
                candidate = _search(item, depth + 1)
                if len(candidate) > len(best):
                    best = candidate
            return best
        return {}

    flat = _search(data)
    if not flat:
        return None

    def _get_val(keys: list[str]):
        """Return (numeric_value, unit_string) for the first matching key."""
        for k in keys:
            for fk, fv in flat.items():
                if k in fk.lower():
                    if isinstance(fv, dict):
                        return _float(fv.get("value")), str(fv.get("unit", "")).lower()
                    return _float(fv), ""
        return None, ""

    ws_v, ws_u = _get_val(["wind-10min-ø", "wind_avg", "windgeschw", "windspd",
                            "wind speed", "windspeed", "ws"])
    wg_v, wg_u = _get_val(["wind-10min-max", "gust", "windböe", "böe", "bö",
                            "wind_gust", "windgust", "wg"])
    wd_v, wd_u = _get_val(["windrichtung", "wind_dir", "winddir", "direction", "wd"])
    tc_v, _    = _get_val(["lufttemp", "temperatur", "temp", "ta"])
    rr_v, _    = _get_val(["niederschlag", "regen", "precip", "rain", "rr"])

    if ws_v is None and tc_v is None:
        return None

    def to_kn(v, u):
        if v is None:
            return None
        if "m/s" in u:
            return _knots_from_ms(v)
        return _knots_from_kmh(v)  # default km/h

    ws_kn  = to_kn(ws_v, ws_u)
    wg_kn  = to_kn(wg_v, wg_u)

    # direction can be numeric degrees or text label
    if wd_v is not None:
        wd_deg = wd_v if isinstance(wd_v, float) else _dir_to_deg(str(wd_v))
    else:
        wd_deg = None
    # try string value from the raw sensor dict if numeric lookup failed
    if wd_deg is None:
        for fk, fv in flat.items():
            if any(tok in fk.lower() for tok in ("windrichtung", "wind_dir", "direction")):
                raw_str = str(fv.get("value", fv) if isinstance(fv, dict) else fv)
                wd_deg = _dir_to_deg(raw_str)
                if wd_deg is not None:
                    break
    wd_txt = _deg_to_compass(wd_deg)

    return {
        "source":        f"wsa-ipsach.meteobase.ch (xhr: {url})",
        "obs_time_utc":  now.isoformat(),
        "hour_key":      hour_key(now),
        "wind_speed_kn": ws_kn,
        "wind_gust_kn":  wg_kn,
        "wind_dir_deg":  wd_deg,
        "wind_dir_txt":  wd_txt,
        "temp_c":        tc_v,
        "precip_mm":     rr_v,
    }


def _parse_wsa_page_text(page_text: str) -> dict | None:
    """
    Parse rendered page text for label-based weather values.
    Handles German labels as found on meteobase.ch station pages.
    """
    now = utc_now()

    def after_label(label: str):
        idx = page_text.find(label)
        if idx == -1:
            return None, None
        snippet = page_text[idx + len(label):idx + len(label) + 80].strip()
        m = re.search(r"([+-]?\d+[.,]?\d*)\s*(km/h|m/s|°[Cc]|mm|%)?", snippet)
        if m:
            return m.group(1).replace(",", "."), (m.group(2) or "").lower()
        w = re.search(r"([A-Za-zÄÖÜäöüß/-]+)", snippet)
        return (w.group(1), "") if w else (None, None)

    def after_label_str(label: str) -> str | None:
        idx = page_text.find(label)
        if idx == -1:
            return None
        snippet = page_text[idx + len(label):idx + len(label) + 40].strip()
        m = re.search(r"([A-Za-z]+)", snippet)
        return m.group(1) if m else None

    def get_num(label: str, *fallbacks: str):
        for lbl in (label, *fallbacks):
            num, unit = after_label(lbl)
            if num is not None:
                return num, unit
        return None, None

    ws_num, ws_unit = get_num("Wind-10min-Ø:", "Wind-10min-O:", "Wind-Ø:",
                               "Windgeschwindigkeit:")
    wg_num, wg_unit = get_num("Wind-10min-Max:", "Windböe:", "Böe:", "Bö:")
    wd_raw = (after_label_str("Windrichtung:") or after_label_str("Richtung:")
              or after_label_str("Wind-Richtung:") or after_label_str("Windrichtung"))
    tc_num, _ = get_num("Lufttemperatur:", "Temperatur:", "Temp.:", "Lufttemp.:")
    rr_num, _ = get_num("Niederschlag:", "Niederschlag 10min:", "Regen:", "Regenmenge:")

    log.info("  WSA text tokens: ws=%s(%s) wg=%s(%s) wd=%s tc=%s rr=%s",
             ws_num, ws_unit, wg_num, wg_unit, wd_raw, tc_num, rr_num)

    def to_knots(num_str, unit_str):
        v = _float(num_str)
        if v is None:
            return None
        return _knots_from_ms(v) if unit_str == "m/s" else _knots_from_kmh(v)

    ws_kn  = to_knots(ws_num, ws_unit)
    wg_kn  = to_knots(wg_num, wg_unit)
    wd_deg = _dir_to_deg(wd_raw)
    wd_txt = _deg_to_compass(wd_deg) or (wd_raw.strip().upper() if wd_raw else None)
    tc     = _float(tc_num)
    rr     = _float(rr_num)

    if ws_kn is None and tc is None:
        return None

    return {
        "source":        "wsa-ipsach.meteobase.ch (page-text)",
        "obs_time_utc":  now.isoformat(),
        "hour_key":      hour_key(now),
        "wind_speed_kn": ws_kn,
        "wind_gust_kn":  wg_kn,
        "wind_dir_deg":  wd_deg,
        "wind_dir_txt":  wd_txt,
        "temp_c":        tc,
        "precip_mm":     rr,
    }


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
    log.info("  Appended measurement → %s", fp)


# ── 3. Collection entry-point ─────────────────────────────────────────────────

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
