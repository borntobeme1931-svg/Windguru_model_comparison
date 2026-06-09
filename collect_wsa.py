#!/usr/bin/env python3
"""
WSA-Ipsach measurement collector
==================================
Fetches the current wind reading from wsa-ipsach.meteobase.ch every 15 minutes
and appends it to a monthly JSONL file.  Also appends a WSA row to the current
hour's snapshot CSV so it stays alongside the Windguru model forecasts.

Run every 15 minutes via GitHub Actions (see .github/workflows/collect.yml).

Dependencies:
    pip install requests playwright
    playwright install chromium
"""

import csv
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# ── configuration ─────────────────────────────────────────────────────────────

DATA_DIR         = Path("weather_data")
MEASUREMENTS_DIR = DATA_DIR / "measurements"
SNAPSHOTS_DIR    = DATA_DIR / "snapshots"

for d in [MEASUREMENTS_DIR, SNAPSHOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(DATA_DIR / "wsa.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def hour_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H")

def minute_key(dt: datetime) -> str:
    """15-minute bucket key, e.g. 2026-06-08T19:45"""
    bucket = (dt.minute // 15) * 15
    return dt.strftime(f"%Y-%m-%dT%H:{bucket:02d}")

def month_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m")

def measurement_file(dt: datetime) -> Path:
    return MEASUREMENTS_DIR / f"{month_key(dt)}.jsonl"

def _float(v) -> float | None:
    try:
        return float(str(v).replace(",", "."))
    except (TypeError, ValueError):
        return None

# Full 16-point compass (German: N=Nord, O=Ost, S=Süd, W=West)
_COMPASS_16 = [
    (  0.0, "N"),   ( 22.5, "NNO"), ( 45.0, "NO"),  ( 67.5, "ONO"),
    ( 90.0, "O"),   (112.5, "OSO"), (135.0, "SO"),   (157.5, "SSO"),
    (180.0, "S"),   (202.5, "SSW"), (225.0, "SW"),   (247.5, "WSW"),
    (270.0, "W"),   (292.5, "WNW"), (315.0, "NW"),   (337.5, "NNW"),
]

_DIR_TO_DEG: dict[str, float] = {}
for _deg, _lbl in _COMPASS_16:
    _DIR_TO_DEG[_lbl.lower()] = _deg
    _DIR_TO_DEG[_lbl.lower().replace("o", "e")] = _deg  # ONO→ENE etc.


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

# ── WSA scraping ──────────────────────────────────────────────────────────────

def _has_wind_data(text: str) -> bool:
    return bool(text) and "Wind-10min" in text


def _fetch_wsa_requests() -> str:
    """
    Fetch WSA page text through a proxy to bypass GitHub Actions IP block.
    Tries scrape.do (free proxy) first, then direct curl_cffi, then plain requests.
    Set SCRAPE_DO_TOKEN env var or repo secret to enable the proxy.
    """
    import os, html as _html, requests as _req

    url = "https://wsa-ipsach.meteobase.ch/"

    def _clean(raw_html: str) -> str:
        text = re.sub(r"<[^>]+>", " ", raw_html)
        text = _html.unescape(text)
        return re.sub(r"[ \t]+", " ", text).strip()

    # attempt 1: scrape.do proxy (routes through residential IP)
    token = os.environ.get("SCRAPE_DO_TOKEN", "")
    if token:
        try:
            proxy_url = f"http://api.scrape.do?token={token}&url={url}&render=false"
            r = _req.get(proxy_url, timeout=30)
            r.raise_for_status()
            text = _clean(r.text)
            log.info("  WSA via scrape.do proxy (%d chars)", len(text))
            return text
        except Exception as exc:
            log.warning("  WSA scrape.do failed: %s", exc)
    else:
        log.debug("  SCRAPE_DO_TOKEN not set, skipping proxy")

    # attempt 2: curl_cffi (TLS fingerprint impersonation)
    try:
        from curl_cffi import requests as cffi_req
        r = cffi_req.get(url, impersonate="chrome124", timeout=15)
        r.raise_for_status()
        text = _clean(r.text)
        log.info("  WSA via curl_cffi (%d chars)", len(text))
        return text
    except ImportError:
        log.debug("  curl_cffi not available")
    except Exception as exc:
        log.warning("  WSA curl_cffi failed: %s", exc)

    # attempt 3: plain requests
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "de-CH,de;q=0.9,en;q=0.8",
        }
        r = _req.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        text = _clean(r.text)
        log.info("  WSA via requests (%d chars)", len(text))
        return text
    except Exception as exc:
        log.warning("  WSA requests failed: %s", exc)
        return ""


def _fetch_wsa_playwright() -> str:
    """Headless browser fallback."""
    if not PLAYWRIGHT_AVAILABLE:
        log.warning("  Playwright not available.")
        return ""

    page_text = ""
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox"],
            )
            page = browser.new_page(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                )
            )
            try:
                page.goto("https://wsa-ipsach.meteobase.ch/",
                          wait_until="domcontentloaded", timeout=30_000)
            except Exception as nav_exc:
                log.warning("  WSA goto warning: %s", nav_exc)

            page.wait_for_timeout(5_000)

            try:
                page_text = page.inner_text("body").strip()
            except Exception:
                try:
                    page_text = page.content()
                    page_text = re.sub(r"<[^>]+>", " ", page_text)
                except Exception:
                    page_text = ""

            browser.close()

    except Exception as exc:
        log.warning("  WSA Playwright error: %s", exc)

    return page_text


def _scrape_wsa() -> dict | None:
    """
    Fetch wsa-ipsach.meteobase.ch and parse wind values from page text.

    Known page format:
        Wind-10min-Ø: 4km/h (2.2kn, 1Bf) SW
        Wind-10min-Max: 10.1km/h (5.5kn, 1Bf) W

    Tries plain HTTP first (fast), falls back to Playwright if blocked.
    """
    page_text = _fetch_wsa_requests()
    if not _has_wind_data(page_text):
        log.info("  Requests fetch missing wind data, trying Playwright …")
        page_text = _fetch_wsa_playwright()

    if not page_text:
        log.warning("  WSA: empty page text.")
        return None

    if not _has_wind_data(page_text):
        log.warning("  WSA: page text has no wind data:\n%s", page_text[:500])
        return None

    log.info("  WSA page (%d chars):\n%s", len(page_text), page_text)

    WIND_RE = re.compile(
        r"Wind-10min-[\u00d8O]:\s*"
        r"[\d.,]+\s*(?:km/h|m/s)"
        r"\s*\(([\d.,]+)\s*kn[^)]*\)"
        r"\s*([A-Z]{1,3})?",
        re.IGNORECASE,
    )
    GUST_RE = re.compile(
        r"Wind-10min-Max:\s*"
        r"[\d.,]+\s*(?:km/h|m/s)"
        r"\s*\(([\d.,]+)\s*kn[^)]*\)",
        re.IGNORECASE,
    )

    ws_kn = wg_kn = wd_deg = wd_txt = None

    m_ws = WIND_RE.search(page_text)
    if m_ws:
        ws_kn  = _float(m_ws.group(1))
        wd_raw = m_ws.group(2)
        wd_deg = _dir_to_deg(wd_raw)
        wd_txt = wd_raw.upper() if wd_raw else None

    m_wg = GUST_RE.search(page_text)
    if m_wg:
        wg_kn = _float(m_wg.group(1))

    log.info("  WSA parsed: wind=%s kn  gust=%s kn  dir=%s (%s°)",
             ws_kn, wg_kn, wd_txt, wd_deg)

    if ws_kn is None:
        log.warning("  WSA: could not extract wind speed from page text.")
        return None

    now = utc_now()
    return {
        "source":        "wsa-ipsach.meteobase.ch",
        "obs_time_utc":  now.isoformat(),
        "hour_key":      hour_key(now),
        "minute_key":    minute_key(now),
        "wind_speed_kn": ws_kn,
        "wind_gust_kn":  wg_kn,
        "wind_dir_deg":  wd_deg,
        "wind_dir_txt":  wd_txt,
        "temp_c":        None,
        "precip_mm":     None,
    }

# ── persistence ───────────────────────────────────────────────────────────────

def save_measurement(obs: dict) -> bool:
    """
    Append to monthly JSONL keyed by 15-minute bucket.
    Returns True if saved, False if already recorded for this bucket.
    """
    now = utc_now()
    fp  = measurement_file(now)
    mk  = obs["minute_key"]

    if fp.exists():
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    if json.loads(line.strip()).get("minute_key") == mk:
                        log.info("  Measurement for %s already recorded — skipping.", mk)
                        return False
                except Exception:
                    pass

    with fp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obs) + "\n")
    log.info("  Appended measurement → %s", fp)
    return True


def update_snapshot(obs: dict, run_time: datetime) -> None:
    """
    Add or update the WSA-Ipsach row in the current hour's snapshot CSV.
    The CSV is created by collect_windguru.py; we just append/replace the WSA row.
    """
    fp = SNAPSHOTS_DIR / f"{run_time.strftime('%Y-%m-%dT%H')}_wsa.csv"
    FIELDS = ["source", "wind_speed_kn", "wind_gust_kn", "wind_dir_deg", "wind_dir_txt"]
    WSA_SOURCE = "WSA-Ipsach (measured)"

    wsa_row = {
        "source":        WSA_SOURCE,
        "wind_speed_kn": obs.get("wind_speed_kn"),
        "wind_gust_kn":  obs.get("wind_gust_kn"),
        "wind_dir_deg":  obs.get("wind_dir_deg"),
        "wind_dir_txt":  obs.get("wind_dir_txt"),
    }

    # Read existing rows (if the Windguru script has already run this hour)
    existing_rows = []
    if fp.exists():
        with fp.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = [r for r in reader if r.get("source") != WSA_SOURCE]

    # Write back with updated WSA row at the end
    with fp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow(wsa_row)

    log.info("  Snapshot updated → %s", fp)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("Fetching WSA-Ipsach measurement …")
    obs = _scrape_wsa()
    if not obs:
        log.error("  ✗ Could not retrieve measurement.")
        sys.exit(1)

    now = utc_now().replace(minute=0, second=0, microsecond=0)
    save_measurement(obs)
    update_snapshot(obs, now)


if __name__ == "__main__":
    main()
