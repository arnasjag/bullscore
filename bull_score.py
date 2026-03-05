#!/usr/bin/env python3
"""
bull_score.py — BullConfidence score (0-100) for BTC-led crypto bull confirmation.

Uses free APIs only: DeFiLlama, CoinGecko/Binance, Binance Futures, mempool.space, FRED.

v3: Z-score adaptive scoring with SQLite history, multi-timeframe DEX, open interest.

Weights rationale (research-calibrated):
  Stablecoins 35% — leading indicator, capital formation at the source
  Macro       20% — leading (10-12wk lag), sets the liquidity ceiling
  Leverage    15% — coincident/contrarian, healthy positioning signal
  Trend       15% — lagging regime filter, confirms but doesn't predict
  Activity    15% — coincident, ecosystem engagement confirmation
"""

import argparse
import hashlib
import json
import math
import os
import random
import sqlite3
import statistics
import sys
import time
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode

import requests

# ─── Constants ────────────────────────────────────────────────────────────────

VERSION = "3.0.0"

URL_STABLECOIN = "https://stablecoins.llama.fi/stablecoincharts/all?stablecoin=1"
URL_DEX = "https://api.llama.fi/overview/dexs?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true"
URL_FEES = "https://api.llama.fi/overview/fees?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true&data=daily&type=fees"
URL_BTC_CG = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
URL_BTC_BINANCE = "https://api.binance.com/api/v3/klines"
URL_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
URL_OI = "https://fapi.binance.com/fapi/v1/openInterest"
URL_MEMPOOL = "https://mempool.space/api/v1/fees/recommended"
URL_FRED = "https://api.stlouisfed.org/fred/series/observations"

# v2/v3 weights: research-calibrated, stablecoins dominant
WEIGHTS = {
    "LiquidityScore": 0.35,
    "MacroScore":     0.20,
    "LeverageScore":  0.15,
    "TrendScore":     0.15,
    "ActivityScore":  0.15,
}

# v1 weights: original spec
WEIGHTS_V1 = {
    "LiquidityScore": 0.30,
    "TrendScore":     0.30,
    "ActivityScore":  0.20,
    "LeverageScore":  0.15,
    "MacroScore":     0.05,
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
CACHE_TTL = 600  # seconds
REQUEST_TIMEOUT = 25
MAX_RETRIES = 3
BASE_DELAY = 2.0
USER_AGENT = "bull-score/3.0"

DB_PATH = os.path.expanduser("~/.bullscore/history.db")
ZSCORE_MIN_SAMPLES = 24  # minimum hourly samples before z-scores activate

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _warn(msg):
    print(f"[WARN] {msg}", file=sys.stderr)


def _info(msg):
    print(f"[INFO] {msg}", file=sys.stderr)


def _clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))


def _sigmoid(x, center, scale):
    """Logistic sigmoid mapping to [0, 1]."""
    z = (x - center) / scale if scale != 0 else 0
    z = max(-20, min(20, z))
    return 1.0 / (1.0 + math.exp(-z))


def _normal_cdf(z):
    """Standard normal CDF using math.erf (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


# ─── Cache ────────────────────────────────────────────────────────────────────


def _cache_key(url):
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _cache_read(url):
    path = os.path.join(CACHE_DIR, _cache_key(url) + ".json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            envelope = json.load(f)
        if time.time() - envelope.get("ts", 0) > CACHE_TTL:
            return None
        return envelope["data"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def _cache_write(url, data):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, _cache_key(url) + ".json")
    try:
        with open(path, "w") as f:
            json.dump({"ts": time.time(), "data": data}, f)
    except OSError:
        pass


# ─── HTTP with retries ───────────────────────────────────────────────────────


def fetch_json(url, params=None, max_retries=MAX_RETRIES, base_delay=BASE_DELAY):
    """Fetch JSON with caching, retries, exponential backoff, jitter, and per-status handling."""
    if params:
        sep = "&" if "?" in url else "?"
        url = url + sep + urlencode(params)

    cached = _cache_read(url)
    if cached is not None:
        return cached

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    short_url = url.split("?")[0]

    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

            if resp.status_code == 200:
                data = resp.json()
                _cache_write(url, data)
                return data

            elif resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", base_delay * (2 ** attempt)))
                retry_after = min(retry_after, 120)
                _warn(f"429 rate-limited {short_url}, waiting {retry_after}s")
                time.sleep(retry_after)
                continue

            elif resp.status_code in (500, 502, 503, 504):
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    _warn(f"{resp.status_code} from {short_url}, retry {attempt+1}/{max_retries} in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                _warn(f"{resp.status_code} from {short_url} after {max_retries} retries")
                return None

            else:
                # Non-retryable (401, 403, 404, etc.)
                _warn(f"HTTP {resp.status_code} from {short_url}")
                return None

        except requests.exceptions.Timeout:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                _warn(f"Timeout {short_url}, retry {attempt+1}/{max_retries} in {delay:.1f}s")
                time.sleep(delay)
                continue
            _warn(f"Timeout {short_url} after {max_retries} retries")
            return None

        except requests.exceptions.ConnectionError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                _warn(f"Connection error {short_url}, retry {attempt+1}/{max_retries} in {delay:.1f}s")
                time.sleep(delay)
                continue
            _warn(f"Connection failed {short_url} after {max_retries} retries")
            return None

        except (requests.RequestException, ValueError) as e:
            _warn(f"Error fetching {short_url}: {e}")
            return None

    return None


# ─── SQLite Storage ──────────────────────────────────────────────────────────


def init_db(db_path=None):
    """Create the snapshots table if it doesn't exist. Returns connection."""
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            source TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL NOT NULL,
            UNIQUE(ts, source, metric)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_source_metric
        ON snapshots(source, metric, ts)
    """)
    conn.commit()
    return conn


def store_snapshot(conn, source, metrics_dict):
    """Insert current hour's metrics. Uses INSERT OR IGNORE to handle duplicates."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")
    count = 0
    for metric, value in metrics_dict.items():
        if value is not None and math.isfinite(value):
            conn.execute(
                "INSERT OR IGNORE INTO snapshots (ts, source, metric, value) VALUES (?, ?, ?, ?)",
                (ts, source, metric, value)
            )
            count += 1
    conn.commit()
    return count


def get_history(conn, source, metric, hours=168):
    """Retrieve last N hours of a metric for z-score calculation."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:00:00Z")
    rows = conn.execute(
        "SELECT value FROM snapshots WHERE source=? AND metric=? AND ts>=? ORDER BY ts",
        (source, metric, cutoff)
    ).fetchall()
    return [r[0] for r in rows]


def get_last_collect_time(conn):
    """Get the most recent snapshot timestamp."""
    row = conn.execute("SELECT MAX(ts) FROM snapshots").fetchone()
    return row[0] if row and row[0] else None


def get_source_freshness(conn, hours=3):
    """Check which sources have recent data."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:00:00Z")
    rows = conn.execute(
        "SELECT DISTINCT source FROM snapshots WHERE ts >= ?", (cutoff,)
    ).fetchall()
    return {r[0] for r in rows}


def get_total_snapshots(conn):
    """Count total rows in snapshots table."""
    row = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()
    return row[0] if row else 0


# ─── Z-Score Engine ──────────────────────────────────────────────────────────


def zscore_to_score(value, history, invert=False):
    """Convert a raw metric to 0-1 score using rolling z-score -> normal CDF.

    Returns None if insufficient history (falls back to v2 clamp scoring).
    """
    if len(history) < ZSCORE_MIN_SAMPLES:
        return None
    mu = statistics.mean(history)
    sigma = statistics.stdev(history)
    if sigma < 1e-12:
        return 0.5
    z = (value - mu) / sigma
    if invert:
        z = -z
    # Cap z at +/- 3 to prevent extreme scores
    z = max(-3.0, min(3.0, z))
    return _normal_cdf(z)


# ─── Data Fetchers ───────────────────────────────────────────────────────────


def fetch_stablecoin_supply():
    """Returns dict with multi-timeframe supply changes, or empty dict on failure.
    Keys: pct_7d, pct_30d, pct_90d, supply_today."""
    data = fetch_json(URL_STABLECOIN)
    if not data or not isinstance(data, list) or len(data) < 91:
        return {}
    try:
        today_val = data[-1]["totalCirculating"]["peggedUSD"]
        result = {"supply_today": today_val}
        for label, offset in [("pct_7d", 7), ("pct_14d", 14), ("pct_30d", 30),
                               ("pct_60d", 60), ("pct_90d", 90)]:
            if len(data) > offset:
                ago_val = data[-(offset + 1)]["totalCirculating"]["peggedUSD"]
                if ago_val > 0:
                    result[label] = (today_val / ago_val) - 1.0
        return result
    except (KeyError, TypeError, IndexError):
        return {}


def fetch_dex_overview():
    """Returns dict with multi-timeframe DEX changes, or empty dict on failure.
    Keys: change_1d, change_7d, change_1m."""
    data = fetch_json(URL_DEX)
    if not data or not isinstance(data, dict):
        return {}
    result = {}
    for key in ("change_1d", "change_7d", "change_1m"):
        val = data.get(key)
        if val is not None:
            try:
                result[key] = float(val)
            except (ValueError, TypeError):
                pass
    return result


def fetch_fees_overview():
    """Returns (fees_accel, raw) or (None, {})."""
    data = fetch_json(URL_FEES)
    if not data or not isinstance(data, dict):
        return None, {}
    try:
        t30 = float(data["total30d"])
        t60to30 = data.get("total60dto30d")
        if t60to30 is not None:
            t60to30 = float(t60to30)
        else:
            t60 = data.get("total60d")
            if t60 is not None:
                t60to30 = float(t60) - t30
        if t60to30 and t60to30 > 0:
            accel = t30 / t60to30
            return accel, {"fees_total30d": round(t30, 2),
                           "fees_total60dto30d": round(t60to30, 2),
                           "fees_accel": round(accel, 4)}
    except (KeyError, TypeError, ValueError):
        pass
    return None, {}


def _parse_btc_prices(prices):
    """Given list of daily prices, compute close/MA50/MA200."""
    if len(prices) < 50:
        return None, None, None, {}
    close = prices[-1]
    ma50 = statistics.mean(prices[-50:])
    ma200 = statistics.mean(prices[-200:]) if len(prices) >= 200 else statistics.mean(prices)
    raw = {"btc_close": round(close, 2), "btc_ma50": round(ma50, 2),
           "btc_ma200": round(ma200, 2), "btc_price_days": len(prices)}
    return close, ma50, ma200, raw


def fetch_btc_prices():
    """Returns (close, ma50, ma200, raw) or (None, None, None, {}).
    Tries CoinGecko first (days=365), falls back to Binance spot klines."""
    data = fetch_json(URL_BTC_CG, params={"vs_currency": "usd", "days": "365"})
    if data and "prices" in data:
        prices = [p[1] for p in data["prices"]
                  if isinstance(p, list) and len(p) == 2]
        if len(prices) >= 200:
            return _parse_btc_prices(prices)
        _warn(f"CoinGecko: only {len(prices)} prices, trying Binance fallback")

    _warn("Using Binance spot klines for BTC prices")
    data = fetch_json(URL_BTC_BINANCE, params={"symbol": "BTCUSDT", "interval": "1d", "limit": "420"})
    if data and isinstance(data, list) and len(data) >= 50:
        prices = [float(candle[4]) for candle in data]
        return _parse_btc_prices(prices)

    return None, None, None, {}


def fetch_funding_rates():
    """Returns (funding_7d_avg, pct_positive_7d, raw) or (None, None, {})."""
    data = fetch_json(URL_FUNDING, params={"symbol": "BTCUSDT", "limit": "1000"})
    if not data or not isinstance(data, list) or len(data) == 0:
        return None, None, {}
    now_ms = time.time() * 1000
    seven_days_ms = 7 * 24 * 3600 * 1000
    recent = []
    for entry in data:
        try:
            ft = int(entry["fundingTime"])
            fr = float(entry["fundingRate"])
            if now_ms - ft <= seven_days_ms:
                recent.append(fr)
        except (KeyError, TypeError, ValueError):
            continue
    if not recent:
        try:
            recent = [float(e["fundingRate"]) for e in data[-21:]]
        except (KeyError, TypeError, ValueError):
            return None, None, {}
    avg = statistics.mean(recent)
    pct_pos = sum(1 for r in recent if r > 0) / len(recent) if recent else 0
    return avg, pct_pos, {"funding_7d_avg": round(avg, 8),
                          "funding_7d_pct_positive": round(pct_pos, 4),
                          "funding_7d_count": len(recent)}


def fetch_open_interest():
    """Returns (oi_value, raw) or (None, {}).
    Fetches current BTCUSDT open interest from Binance Futures."""
    data = fetch_json(URL_OI, params={"symbol": "BTCUSDT"})
    if not data or not isinstance(data, dict):
        return None, {}
    try:
        oi = float(data["openInterest"])
        return oi, {"oi_btc": round(oi, 2), "oi_time": data.get("time")}
    except (KeyError, ValueError, TypeError):
        return None, {}


def fetch_mempool_fees():
    """Returns (fastestFee, raw) or (None, {})."""
    data = fetch_json(URL_MEMPOOL)
    if not data or not isinstance(data, dict):
        return None, {}
    fee = data.get("fastestFee")
    if fee is not None:
        try:
            fee = float(fee)
            return fee, {"mempool_fastest_fee": fee}
        except (ValueError, TypeError):
            pass
    return None, {}


def fetch_fred_walcl(api_key):
    """Returns (walcl_13w_pct, raw) or (None, {})."""
    if not api_key:
        return None, {}
    data = fetch_json(URL_FRED, params={
        "series_id": "WALCL",
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": "200",
    })
    if not data or "observations" not in data:
        return None, {}
    obs = data["observations"]
    valid = []
    for o in obs:
        try:
            val = float(o["value"])
            valid.append((o["date"], val))
        except (KeyError, ValueError, TypeError):
            continue
    if len(valid) < 14:
        return None, {}
    w_today = valid[0][1]
    w_13w = valid[min(13, len(valid) - 1)][1]
    if w_13w > 0:
        pct = (w_today / w_13w) - 1.0
        return pct, {"walcl_today": w_today, "walcl_13w_ago": w_13w,
                     "walcl_13w_pct": round(pct, 6)}
    return None, {}


# ─── Component Scorers ───────────────────────────────────────────────────────
#
# Each scorer has two paths:
#   1. Z-score path (v3): uses historical distribution when enough data exists
#   2. Clamp path (v2 fallback): fixed ranges, works from first run


def score_liquidity(stable_data, mode="standard", conn=None):
    """LiquidityScore from multi-timeframe stablecoin supply changes.

    v3: Z-score of each timeframe when history available, else v2 clamp.
    v2: 7d (20%), 30d (50%), 90d (30%) with fixed clamp ranges.
    """
    if not stable_data or "pct_30d" not in stable_data:
        return None

    raw = {}
    sub_scores = []
    sub_weights = []

    timeframes = [
        ("pct_7d",  0.20, 0.005, 0.02),    # key, weight, clamp_offset, clamp_range
        ("pct_30d", 0.50, 0.02,  0.06),
        ("pct_90d", 0.30, 0.03,  0.11),
    ]

    for key, weight, clamp_off, clamp_range in timeframes:
        val = stable_data.get(key)
        if val is None:
            continue
        raw[f"stable_{key}"] = round(val * 100, 3)

        if mode == "v1" and key != "pct_30d":
            continue
        if mode == "v1":
            return _clamp((val + 0.02) / 0.06), raw

        # Try z-score first
        z_score = None
        if conn is not None:
            history = get_history(conn, "stablecoin", key)
            z_score = zscore_to_score(val, history)

        if z_score is not None:
            sub_scores.append(z_score)
        else:
            # v2 fallback: fixed clamp
            sub_scores.append(_clamp((val + clamp_off) / clamp_range))
        sub_weights.append(weight)

    raw["stable_supply_B"] = round(stable_data.get("supply_today", 0) / 1e9, 2)

    if not sub_scores:
        return None
    total_w = sum(sub_weights)
    score = sum(s * w for s, w in zip(sub_scores, sub_weights)) / total_w
    return round(score, 4), raw


def score_trend(close, ma50, ma200, mode="standard"):
    """TrendScore from BTC price relative to moving averages.

    Kept as continuous ratio-based (v2) — z-scores don't help here since
    MA ratios are already normalized. Trend is a lagging confirmation signal.
    """
    if close is None or ma200 is None:
        return None

    raw = {}
    ratio_200 = close / ma200 if ma200 > 0 else 1.0
    raw["btc_close"] = round(close, 2)
    raw["btc_ma200"] = round(ma200, 2)
    raw["close_to_ma200"] = round(ratio_200, 4)

    if mode == "v1":
        above_200 = 1.0 if close > ma200 else 0.0
        golden = (1.0 if ma50 > ma200 else 0.0) if ma50 else 0.5
        return 0.5 * above_200 + 0.5 * golden, raw

    # v2/v3: continuous scoring
    s_200 = _clamp((ratio_200 - 0.75) / 0.50)

    s_golden = 0.5
    if ma50 is not None:
        raw["btc_ma50"] = round(ma50, 2)
        ratio_50_200 = ma50 / ma200 if ma200 > 0 else 1.0
        raw["ma50_to_ma200"] = round(ratio_50_200, 4)
        s_golden = _clamp((ratio_50_200 - 0.85) / 0.30)

    s_short = 0.5
    if ma50 is not None:
        ratio_close_50 = close / ma50 if ma50 > 0 else 1.0
        raw["close_to_ma50"] = round(ratio_close_50, 4)
        s_short = _clamp((ratio_close_50 - 0.90) / 0.20)

    score = 0.40 * s_200 + 0.30 * s_golden + 0.30 * s_short
    return round(score, 4), raw


def score_activity(dex_data, fees_accel, fee_pressure, mode="standard", conn=None):
    """ActivityScore from multi-timeframe DEX volume, fee acceleration, mempool fees.

    v3: Z-scores of each DEX timeframe when history available, else v2 clamp.
    v2: DEX 50%, fees 35%, mempool 15%.
    """
    subs = []
    sub_weights = []
    raw = {}

    # Multi-timeframe DEX volume changes
    if isinstance(dex_data, dict):
        dex_timeframes = [
            ("change_1d", 0.20, 10, 30),   # key, weight, clamp_offset, clamp_range
            ("change_7d", 0.25, 15, 45),
            ("change_1m", 0.25, 20, 60),
        ]
        for key, weight, clamp_off, clamp_range in dex_timeframes:
            val = dex_data.get(key)
            if val is None:
                continue
            raw[f"dex_{key}"] = round(val, 2)

            if mode == "v1":
                if key == "change_1m":
                    subs.append(_clamp((val + 20) / 60))
                    sub_weights.append(1)
                continue

            z_score = None
            if conn is not None:
                history = get_history(conn, "dex", key)
                z_score = zscore_to_score(val, history)

            if z_score is not None:
                subs.append(z_score)
            else:
                subs.append(_clamp((val + clamp_off) / clamp_range))
            sub_weights.append(weight)

    elif dex_data is not None and not isinstance(dex_data, dict):
        # Legacy single-value compat
        raw["dex_change_1m"] = round(dex_data, 2)
        subs.append(_clamp((dex_data + 20) / 60))
        sub_weights.append(0.50 if mode != "v1" else 1)

    if fees_accel is not None:
        raw["fees_accel"] = round(fees_accel, 4)
        if mode == "v1":
            subs.append(_clamp((fees_accel - 0.8) / 0.4))
            sub_weights.append(1)
        else:
            z_score = None
            if conn is not None:
                history = get_history(conn, "fees", "accel")
                z_score = zscore_to_score(fees_accel, history)
            if z_score is not None:
                subs.append(z_score)
            else:
                subs.append(_clamp((fees_accel - 0.7) / 0.6))
            sub_weights.append(0.20)

    if fee_pressure is not None:
        raw["mempool_fastest_fee"] = fee_pressure
        if mode == "v1":
            subs.append(_clamp((fee_pressure - 5) / 50))
            sub_weights.append(1)
        else:
            log_fee = math.log(max(fee_pressure, 1))
            log_low = math.log(3)
            log_high = math.log(50)
            subs.append(_clamp((log_fee - log_low) / (log_high - log_low)))
            sub_weights.append(0.10)

    if not subs:
        return None

    if mode == "v1":
        return round(statistics.mean(subs), 4), raw

    total_w = sum(sub_weights)
    score = sum(s * w for s, w in zip(subs, sub_weights)) / total_w
    return round(score, 4), raw


def score_leverage(funding_avg, pct_positive, oi_value=None, mode="standard", conn=None):
    """LeverageScore from funding rates + open interest.

    v3: Z-score of funding rate + OI when history available.
    v2: Smooth piecewise funding curve + % positive blend.
    OI adds signal: rising OI + positive funding = stronger conviction.
    """
    if funding_avg is None:
        return None

    f = funding_avg
    raw = {"funding_7d_avg": round(f, 8)}
    if pct_positive is not None:
        raw["funding_7d_pct_positive"] = round(pct_positive, 4)
    if oi_value is not None:
        raw["oi_btc"] = round(oi_value, 2)

    if mode == "v1":
        if f < 0:
            return 0.0, raw
        if f <= 0.0002:
            return round(f / 0.0002, 4), raw
        return round(max(0.0, 1.0 - (f - 0.0002) / 0.0004), 4), raw

    # v2/v3 funding rate scoring
    z_funding = None
    if conn is not None:
        history = get_history(conn, "funding", "avg_7d")
        z_funding = zscore_to_score(f, history)

    if z_funding is not None:
        s_rate = z_funding
    else:
        # v2 fallback: smooth piecewise
        if f <= -0.0003:
            s_rate = 0.0
        elif f <= 0:
            s_rate = 0.15 + 0.20 * ((f + 0.0003) / 0.0003)
        elif f <= 0.00005:
            s_rate = 0.35 + 0.15 * (f / 0.00005)
        elif f <= 0.0002:
            s_rate = 0.50 + 0.45 * ((f - 0.00005) / 0.00015)
        elif f <= 0.0003:
            s_rate = 0.95 - 0.15 * ((f - 0.0002) / 0.0001)
        else:
            s_rate = max(0.0, 0.80 - (f - 0.0003) / 0.0003)

    # OI signal (20% of leverage score)
    s_oi = None
    if oi_value is not None and conn is not None:
        history = get_history(conn, "oi", "open_interest")
        s_oi = zscore_to_score(oi_value, history)

    # % positive funding (secondary signal)
    s_pos = None
    if pct_positive is not None:
        s_pos = _clamp((pct_positive - 0.30) / 0.70)

    # Blend: adaptive weights based on what's available
    if s_oi is not None and s_pos is not None:
        score = 0.50 * s_rate + 0.20 * s_oi + 0.30 * s_pos
    elif s_oi is not None:
        score = 0.65 * s_rate + 0.35 * s_oi
    elif s_pos is not None:
        score = 0.70 * s_rate + 0.30 * s_pos
    else:
        score = s_rate

    return round(_clamp(score), 4), raw


def score_macro(walcl_pct, mode="standard", conn=None):
    """MacroScore from FRED WALCL 13w change."""
    if walcl_pct is None:
        return None

    raw = {"walcl_13w_pct": round(walcl_pct * 100, 3)}

    if mode == "v1":
        return round(_clamp((walcl_pct + 0.01) / 0.02), 4), raw

    # Try z-score
    z_score = None
    if conn is not None:
        history = get_history(conn, "fred", "walcl_13w_pct")
        z_score = zscore_to_score(walcl_pct, history)

    if z_score is not None:
        return round(z_score, 4), raw

    # v2 fallback: sigmoid
    return round(_clamp(_sigmoid(walcl_pct, center=0.0, scale=0.008)), 4), raw


# ─── Collection (--collect) ──────────────────────────────────────────────────


def collect_all(conn, fred_key=None):
    """Fetch all APIs and store snapshots in SQLite. Returns summary dict."""
    metrics_stored = 0
    sources_ok = []
    sources_failed = []

    # 1. Stablecoins
    stable_data = fetch_stablecoin_supply()
    if stable_data and "pct_30d" in stable_data:
        stored = store_snapshot(conn, "stablecoin", {
            k: v for k, v in stable_data.items() if k != "supply_today"
        })
        if "supply_today" in stable_data:
            store_snapshot(conn, "stablecoin", {"supply_today": stable_data["supply_today"]})
            stored += 1
        metrics_stored += stored
        sources_ok.append("stablecoin")
    else:
        sources_failed.append("stablecoin")

    # 2. DEX (multi-timeframe)
    dex_data = fetch_dex_overview()
    if dex_data:
        metrics_stored += store_snapshot(conn, "dex", dex_data)
        sources_ok.append("dex")
    else:
        sources_failed.append("dex")

    # 3. Fees
    fees_accel, fees_raw = fetch_fees_overview()
    if fees_accel is not None:
        metrics_stored += store_snapshot(conn, "fees", {"accel": fees_accel})
        sources_ok.append("fees")
    else:
        sources_failed.append("fees")

    # 4. BTC prices
    close, ma50, ma200, btc_raw = fetch_btc_prices()
    if close is not None:
        btc_metrics = {"close": close}
        if ma50 is not None:
            btc_metrics["ma50"] = ma50
        if ma200 is not None:
            btc_metrics["ma200"] = ma200
            btc_metrics["close_to_ma200"] = close / ma200 if ma200 > 0 else 1.0
        if ma50 is not None and ma200 is not None and ma200 > 0:
            btc_metrics["ma50_to_ma200"] = ma50 / ma200
        metrics_stored += store_snapshot(conn, "btc", btc_metrics)
        sources_ok.append("btc")
    else:
        sources_failed.append("btc")

    # 5. Funding rates
    funding_avg, pct_positive, funding_raw = fetch_funding_rates()
    if funding_avg is not None:
        fund_metrics = {"avg_7d": funding_avg}
        if pct_positive is not None:
            fund_metrics["pct_positive"] = pct_positive
        metrics_stored += store_snapshot(conn, "funding", fund_metrics)
        sources_ok.append("funding")
    else:
        sources_failed.append("funding")

    # 6. Open interest
    oi_value, oi_raw = fetch_open_interest()
    if oi_value is not None:
        metrics_stored += store_snapshot(conn, "oi", {"open_interest": oi_value})
        sources_ok.append("oi")
    else:
        sources_failed.append("oi")

    # 7. Mempool fees
    fee_pressure, mem_raw = fetch_mempool_fees()
    if fee_pressure is not None:
        metrics_stored += store_snapshot(conn, "mempool", {"fastest_fee": fee_pressure})
        sources_ok.append("mempool")
    else:
        sources_failed.append("mempool")

    # 8. FRED WALCL (optional)
    walcl_pct, walcl_raw = fetch_fred_walcl(fred_key)
    if walcl_pct is not None:
        metrics_stored += store_snapshot(conn, "fred", {"walcl_13w_pct": walcl_pct})
        sources_ok.append("fred")
    else:
        sources_failed.append("fred")

    return {
        "metrics_stored": metrics_stored,
        "sources_ok": sources_ok,
        "sources_failed": sources_failed,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ─── Aggregation ─────────────────────────────────────────────────────────────


def compute_score(mode="standard", fred_key=None, conn=None):
    """Fetch all data, compute components, aggregate into final score.

    v3: Uses z-score scoring when enough history is available in SQLite.
    Falls back to v2 clamp scoring when history is insufficient.
    """
    raw_features = {}
    weights = WEIGHTS_V1 if mode == "v1" else WEIGHTS

    # ── Fetch all data ──
    stable_data = fetch_stablecoin_supply()

    dex_data = fetch_dex_overview()
    if isinstance(dex_data, dict):
        for k, v in dex_data.items():
            raw_features[f"dex_{k}"] = round(v, 4)

    fees_accel, raw = fetch_fees_overview()
    raw_features.update(raw)

    close, ma50, ma200, raw = fetch_btc_prices()
    raw_features.update(raw)

    funding_avg, pct_positive, raw = fetch_funding_rates()
    raw_features.update(raw)

    oi_value, raw = fetch_open_interest()
    raw_features.update(raw)

    fee_pressure, raw = fetch_mempool_fees()
    raw_features.update(raw)

    walcl_pct, raw = fetch_fred_walcl(fred_key)
    raw_features.update(raw)

    # ── Score each component ──
    components = {}
    available = {}

    # LiquidityScore
    result = score_liquidity(stable_data, mode, conn)
    if result is None:
        _warn("LiquidityScore: data unavailable, redistributing weight")
        components["LiquidityScore"] = None
    else:
        score, raw = result
        components["LiquidityScore"] = score
        available["LiquidityScore"] = score
        raw_features.update(raw)

    # TrendScore
    result = score_trend(close, ma50, ma200, mode)
    if result is None:
        _warn("TrendScore: data unavailable, redistributing weight")
        components["TrendScore"] = None
    else:
        score, raw = result
        components["TrendScore"] = score
        available["TrendScore"] = score
        raw_features.update(raw)

    # ActivityScore
    result = score_activity(dex_data, fees_accel, fee_pressure, mode, conn)
    if result is None:
        _warn("ActivityScore: data unavailable, redistributing weight")
        components["ActivityScore"] = None
    else:
        score, raw = result
        components["ActivityScore"] = score
        available["ActivityScore"] = score
        raw_features.update(raw)

    # LeverageScore
    result = score_leverage(funding_avg, pct_positive, oi_value, mode, conn)
    if result is None:
        _warn("LeverageScore: data unavailable, redistributing weight")
        components["LeverageScore"] = None
    else:
        score, raw = result
        components["LeverageScore"] = score
        available["LeverageScore"] = score
        raw_features.update(raw)

    # MacroScore
    result = score_macro(walcl_pct, mode, conn)
    if result is None:
        if mode == "v1":
            components["MacroScore"] = 0.5
            available["MacroScore"] = 0.5
        else:
            _warn("MacroScore: FRED data unavailable, redistributing weight")
            components["MacroScore"] = None
    else:
        score, raw = result
        components["MacroScore"] = score
        available["MacroScore"] = score
        raw_features.update(raw)

    # ── Weighted aggregation with redistribution ──
    if not available:
        bull_confidence = 50
    else:
        total_available_weight = sum(weights[k] for k in available)
        if total_available_weight > 0:
            bull_raw = sum((weights[k] / total_available_weight) * available[k]
                          for k in available)
        else:
            bull_raw = 0.5
        bull_confidence = int(round(bull_raw * 100))
        bull_confidence = max(0, min(100, bull_confidence))

    effective_weights = {}
    total_available_weight = sum(weights[k] for k in available) if available else 1
    for k in weights:
        if k in available and total_available_weight > 0:
            effective_weights[k] = round(weights[k] / total_available_weight, 4)
        else:
            effective_weights[k] = 0.0

    # Detect scoring mode
    scoring_mode = "v2-clamp"
    if mode != "v1" and conn is not None:
        sample_hist = get_history(conn, "stablecoin", "pct_30d")
        if len(sample_hist) >= ZSCORE_MIN_SAMPLES:
            scoring_mode = "v3-zscore"

    return {
        "bull_confidence": bull_confidence,
        "version": VERSION,
        "mode": mode,
        "scoring": scoring_mode,
        "component_scores": components,
        "effective_weights": effective_weights,
        "raw_features": raw_features,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ─── Delivery (--deliver) ───────────────────────────────────────────────────


def format_telegram_message(result):
    """Format score result as a Telegram-friendly text message."""
    score = result["bull_confidence"]
    cs = result["component_scores"]
    rf = result["raw_features"]

    # Bull/bear emoji
    if score >= 70:
        emoji = "\U0001f7e2"  # green circle
    elif score >= 40:
        emoji = "\U0001f7e1"  # yellow circle
    else:
        emoji = "\U0001f534"  # red circle

    lines = [f"{emoji} BullConfidence: {score}/100"]
    lines.append("")

    # Component breakdown
    parts = []
    for name in ["LiquidityScore", "TrendScore", "ActivityScore", "LeverageScore", "MacroScore"]:
        val = cs.get(name)
        short = name.replace("Score", "")
        if val is not None:
            parts.append(f"{short}: {int(val*100)}%")
        else:
            parts.append(f"{short}: n/a")
    lines.append(" | ".join(parts[:3]))
    lines.append(" | ".join(parts[3:]))
    lines.append("")

    # Key insights
    insights = []
    p30 = rf.get("stable_pct_30d")
    if p30 is not None:
        direction = "growing" if p30 > 0 else "contracting"
        insights.append(f"Stablecoins {direction} ({p30:+.1f}% 30d)")

    ratio = rf.get("close_to_ma200")
    if ratio is not None:
        pct = (ratio - 1) * 100
        pos = "above" if pct >= 0 else "below"
        insights.append(f"BTC {abs(pct):.0f}% {pos} MA200")

    favg = rf.get("funding_7d_avg")
    if favg is not None:
        if abs(favg) < 0.00005:
            insights.append("Funding neutral")
        elif favg > 0:
            insights.append(f"Funding positive ({favg*100:.3f}%)")
        else:
            insights.append(f"Funding negative ({favg*100:.3f}%)")

    if insights:
        lines.append("Key: " + " | ".join(insights))

    lines.append("")
    lines.append(f"Scoring: {result.get('scoring', 'v2-clamp')} | {result['timestamp_utc'][:16]}")

    return "\n".join(lines)


def deliver_to_openclaw(message, webhook_url="http://127.0.0.1:18789/hooks/agent", token=None):
    """POST formatted message to OpenClaw webhook for Telegram delivery."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = {
        "message": message,
        "name": "BullScore",
        "deliver": True,
        "channel": "telegram",
    }

    try:
        resp = requests.post(webhook_url, json=payload, headers=headers, timeout=15)
        if resp.status_code in (200, 202):
            _info(f"Delivered to OpenClaw (HTTP {resp.status_code})")
            return True
        else:
            _warn(f"OpenClaw webhook returned HTTP {resp.status_code}: {resp.text[:200]}")
            return False
    except requests.RequestException as e:
        _warn(f"OpenClaw delivery failed: {e}")
        return False


# ─── Health Check (--health) ─────────────────────────────────────────────────


def health_check(conn):
    """Return health status JSON."""
    all_sources = ["stablecoin", "dex", "fees", "btc", "funding", "oi", "mempool", "fred"]

    last_ts = get_last_collect_time(conn)
    total = get_total_snapshots(conn)
    fresh = get_source_freshness(conn, hours=3)

    # Determine staleness
    age_minutes = None
    if last_ts:
        try:
            last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
            age_minutes = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60
        except (ValueError, TypeError):
            pass

    sources_ok = [s for s in all_sources if s in fresh]
    sources_stale = [s for s in all_sources if s not in fresh and s != "fred"]
    sources_missing = [s for s in all_sources if s not in fresh and s == "fred"]

    if age_minutes is None or age_minutes > 360:
        status = "unhealthy"
    elif age_minutes > 120:
        status = "degraded"
    else:
        status = "healthy"

    # DB size
    db_path = conn.execute("PRAGMA database_list").fetchone()
    db_size_mb = 0
    if db_path and db_path[2]:
        try:
            db_size_mb = round(os.path.getsize(db_path[2]) / (1024 * 1024), 2)
        except OSError:
            pass

    return {
        "status": status,
        "last_collect_utc": last_ts,
        "last_collect_age_minutes": round(age_minutes, 1) if age_minutes else None,
        "total_snapshots": total,
        "sources_ok": sources_ok,
        "sources_stale": sources_stale,
        "sources_missing": sources_missing,
        "db_size_mb": db_size_mb,
    }


# ─── Self-Test ───────────────────────────────────────────────────────────────


def self_test(fred_key=None, db_path=None):
    """Hit each endpoint, validate schema, check score bounds, test DB."""
    tests = []
    ok_count = 0

    def check(name, fn, validator):
        nonlocal ok_count
        try:
            result = fn()
            if validator(result):
                tests.append((name, "PASS", "ok"))
                ok_count += 1
            else:
                tests.append((name, "FAIL", "validation failed"))
        except Exception as e:
            tests.append((name, "FAIL", str(e)))

    check("stablecoin_supply", fetch_stablecoin_supply,
          lambda r: isinstance(r, dict) and "pct_30d" in r)
    check("dex_overview", fetch_dex_overview,
          lambda r: isinstance(r, dict) and len(r) > 0)
    check("fees_overview", fetch_fees_overview,
          lambda r: r[0] is not None and math.isfinite(r[0]))
    check("btc_prices", fetch_btc_prices,
          lambda r: r[0] is not None and r[2] is not None and math.isfinite(r[0]))
    check("funding_rates", fetch_funding_rates,
          lambda r: r[0] is not None and math.isfinite(r[0]))
    check("open_interest", fetch_open_interest,
          lambda r: r[0] is not None and r[0] > 0)
    check("mempool_fees", fetch_mempool_fees,
          lambda r: r[0] is not None and math.isfinite(r[0]))
    check("fred_walcl", lambda: fetch_fred_walcl(fred_key),
          lambda r: True)  # optional

    # DB test
    try:
        test_conn = init_db(db_path)
        store_snapshot(test_conn, "_test", {"_test_metric": 42.0})
        hist = get_history(test_conn, "_test", "_test_metric", hours=1)
        test_conn.execute("DELETE FROM snapshots WHERE source='_test'")
        test_conn.commit()
        if len(hist) > 0 and hist[0] == 42.0:
            tests.append(("sqlite_db", "PASS", "ok"))
            ok_count += 1
        else:
            tests.append(("sqlite_db", "FAIL", f"round-trip failed: {hist}"))
        test_conn.close()
    except Exception as e:
        tests.append(("sqlite_db", "FAIL", str(e)))

    # Score bounds check
    try:
        result = compute_score(fred_key=fred_key)
        score = result["bull_confidence"]
        score_ok = 0 <= score <= 100
        for k, v in result["component_scores"].items():
            if v is not None and not (0.0 <= v <= 1.0):
                score_ok = False
        if score_ok:
            tests.append(("score_bounds", "PASS", f"bull_confidence={score}"))
            ok_count += 1
        else:
            tests.append(("score_bounds", "FAIL", f"out of range: {score}"))
    except Exception as e:
        tests.append(("score_bounds", "FAIL", str(e)))

    print("--- Self-Test Results ---", file=sys.stderr)
    for name, status, msg in tests:
        print(f"  {status} {name}: {msg}", file=sys.stderr)
    print(f"  {ok_count}/{len(tests)} passed", file=sys.stderr)

    data_passes = sum(1 for name, status, _ in tests[:7] if status == "PASS")
    if data_passes == 0:
        print("FAIL: all data sources failed", file=sys.stderr)
        return 1
    return 0


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="BullConfidence score (0-100) for BTC-led crypto bull confirmation."
    )
    parser.add_argument("--self-test", action="store_true",
                        help="Validate endpoints, DB, and score bounds.")
    parser.add_argument("--collect", action="store_true",
                        help="Fetch all APIs and store snapshots in SQLite.")
    parser.add_argument("--deliver", action="store_true",
                        help="Collect, compute score, and send to OpenClaw/Telegram.")
    parser.add_argument("--health", action="store_true",
                        help="Check health status of collection pipeline.")
    parser.add_argument("--mode", choices=["standard", "v1"], default="standard",
                        help="Scoring mode: standard (v3 z-score / v2 fallback) or v1 (original).")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass local file cache.")
    parser.add_argument("--db", default=None,
                        help=f"SQLite DB path (default: {DB_PATH}).")
    parser.add_argument("--webhook-url", default="http://127.0.0.1:18789/hooks/agent",
                        help="OpenClaw webhook URL for --deliver.")
    parser.add_argument("--webhook-token", default=None,
                        help="Bearer token for OpenClaw webhook.")
    args = parser.parse_args()

    fred_key = os.environ.get("FRED_API_KEY", "")
    webhook_token = args.webhook_token or os.environ.get("OPENCLAW_TOKEN", "")

    if args.no_cache:
        global CACHE_TTL
        CACHE_TTL = 0

    db_path = args.db or DB_PATH

    if args.self_test:
        sys.exit(self_test(fred_key, db_path))

    if args.health:
        conn = init_db(db_path)
        result = health_check(conn)
        conn.close()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["status"] != "unhealthy" else 1)

    if args.collect:
        conn = init_db(db_path)
        summary = collect_all(conn, fred_key)
        ts = summary["timestamp_utc"][:16]
        ok = len(summary["sources_ok"])
        fail = len(summary["sources_failed"])
        _info(f"[{ts}] collected {summary['metrics_stored']} metrics from {ok} sources, {fail} failures")
        if summary["sources_failed"]:
            _warn(f"Failed: {', '.join(summary['sources_failed'])}")
        conn.close()
        if not args.deliver:
            sys.exit(0 if ok > 0 else 1)

    if args.deliver:
        conn = init_db(db_path)
        if not args.collect:
            # collect first if not already done
            collect_all(conn, fred_key)
        result = compute_score(mode=args.mode, fred_key=fred_key, conn=conn)
        conn.close()
        message = format_telegram_message(result)
        print(message, file=sys.stderr)
        delivered = deliver_to_openclaw(message, args.webhook_url, webhook_token)
        # Also print JSON to stdout
        print(json.dumps(result, indent=2))
        sys.exit(0 if delivered else 1)

    # Default: compute and print score
    conn = None
    if os.path.exists(db_path) or os.path.exists(os.path.dirname(db_path)):
        try:
            conn = init_db(db_path)
        except Exception:
            pass

    result = compute_score(mode=args.mode, fred_key=fred_key, conn=conn)
    if conn:
        conn.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
