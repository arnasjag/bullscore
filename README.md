# BullConfidence Score

A single-file Python script that computes a **BullConfidence score (0-100)** for BTC-led crypto bull confirmation using free APIs and wall street math (z-scores).

## Architecture

```
bull_score.py
├── --collect       Hourly: fetch 8 APIs → store snapshot in SQLite
├── --report        Compute z-score/clamp score → JSON to stdout
├── --deliver       Collect + compute + send to Telegram via OpenClaw
├── --health        Pipeline health check (JSON)
├── --self-test     Validate all endpoints + DB + score bounds
├── --mode v1       Legacy v1 binary scoring
└── (default)       Same as --report
```

## Install

```bash
pip install requests
```

Python 3.8+ required. No other dependencies (sqlite3 is stdlib).

## Required Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FRED_API_KEY` | No | FRED API key for Fed balance sheet data. Get one free at https://fred.stlouisfed.org/docs/api/api_key.html. If absent, MacroScore weight is redistributed. |
| `OPENCLAW_TOKEN` | No | Bearer token for OpenClaw webhook (used by `--deliver`). |

## Usage

```bash
# Compute today's score (v3 z-score when history available, v2 clamp fallback)
python bull_score.py

# Collect data and store in SQLite (run hourly via cron/LaunchAgent)
python bull_score.py --collect

# Collect + compute + deliver to Telegram via OpenClaw
python bull_score.py --deliver

# Check pipeline health
python bull_score.py --health

# Self-test: validate all endpoints, DB, and score bounds
python bull_score.py --self-test

# Legacy v1 scoring (binary MA cross, equal-ish weights)
python bull_score.py --mode v1

# Skip cache (fresh API calls)
python bull_score.py --no-cache

# Custom DB path
python bull_score.py --db /path/to/history.db
```

## Endpoints Used

| # | Source | Endpoint | Data Extracted |
|---|--------|----------|----------------|
| 1 | DeFiLlama | `stablecoins.llama.fi/stablecoincharts/all?stablecoin=1` | Stablecoin supply time series → 7d/30d/90d % changes |
| 2 | DeFiLlama | `api.llama.fi/overview/dexs` | DEX volume change_1d, change_7d, change_1m |
| 3 | DeFiLlama | `api.llama.fi/overview/fees` | Protocol fees `total30d` / `total60dto30d` → fee acceleration |
| 4 | CoinGecko / Binance | BTC daily prices (365d CoinGecko, 420d Binance fallback) | close, MA50, MA200, ratios |
| 5 | Binance Futures | `fapi.binance.com/fapi/v1/fundingRate` | BTCUSDT funding rate → 7d avg + % positive periods |
| 6 | Binance Futures | `fapi.binance.com/fapi/v1/openInterest` | BTCUSDT open interest (BTC notional) |
| 7 | mempool.space | `mempool.space/api/v1/fees/recommended` | Bitcoin `fastestFee` (sat/vB) |
| 8 | FRED | `api.stlouisfed.org/fred/series/observations` (WALCL) | Fed balance sheet → 13-week % change |

## Scoring (v3)

```
BullConfidence = 100 × (0.35×Liquidity + 0.20×Macro + 0.15×Leverage + 0.15×Trend + 0.15×Activity)
```

### Z-Score Adaptive Scoring

When enough history exists (24+ hourly snapshots), metrics are scored using rolling z-scores mapped through the normal CDF:

```
z = (value - mean) / stdev
score = Φ(z)    # normal CDF, [0, 1]
```

This adapts automatically to market regime changes. When history is insufficient, falls back to v2 fixed-range clamp scoring.

### Weight Rationale

| Component | Weight | Type | Why |
|-----------|--------|------|-----|
| LiquidityScore | **35%** | **Leading** | Stablecoin supply growth is the strongest on-chain predictor. Capital enters via stablecoins before flowing into BTC/alts. |
| MacroScore | **20%** | **Leading** (10-12wk lag) | Fed balance sheet expansion sets the liquidity ceiling. BTC follows global M2 with ~70-90 day lag. |
| LeverageScore | **15%** | **Coincident** | Funding rates + open interest show real-time positioning. Rising OI + positive funding = stronger conviction. |
| TrendScore | **15%** | **Lagging** | MA cross confirms trend regime but doesn't predict. 73% hit rate historically. |
| ActivityScore | **15%** | **Coincident** | Multi-timeframe DEX volume + protocol fees confirm ecosystem engagement. |

### Component Details

**LiquidityScore** — Multi-timeframe stablecoin supply:
- 20%: 7d momentum (z-score or clamp `[-0.5%, +1.5%]`)
- 50%: 30d trend (z-score or clamp `[-2%, +4%]`)
- 30%: 90d regime (z-score or clamp `[-3%, +8%]`)

**TrendScore** — Continuous ratio-based (always v2, z-scores don't help here):
- 40%: price/MA200 ratio (0.75→0, 1.0→0.5, 1.25→1.0)
- 30%: MA50/MA200 ratio (0.85→0, 1.0→0.5, 1.15→1.0)
- 30%: price/MA50 ratio (0.90→0, 1.0→0.5, 1.10→1.0)

**ActivityScore** — Multi-timeframe DEX + fees:
- 20%: DEX change_1d (z-score or clamp)
- 25%: DEX change_7d (z-score or clamp)
- 25%: DEX change_1m (z-score or clamp)
- 20%: Fee acceleration (z-score or clamp)
- 10%: Mempool fees (log-scaled)

**LeverageScore** — Funding rate + open interest:
- 50%: Funding rate (z-score or smooth piecewise)
- 20%: Open interest (z-score, when history available)
- 30%: % positive funding periods

**MacroScore** — WALCL 13w change (z-score or sigmoid, center=0, scale=0.8%)

## Retry Logic

All API calls use exponential backoff with jitter:
- 3 retries max, 2s base delay
- 429: respects `Retry-After` header, capped at 120s
- 500/502/503/504: retries with `delay * 2^attempt + jitter`
- 401/403/404: fails immediately (no retry)
- Timeout/ConnectionError: retries with backoff

## Storage

SQLite database at `~/.bullscore/history.db`. Stores hourly snapshots of all metrics with `UNIQUE(ts, source, metric)` constraint (duplicate inserts ignored).

Use `--collect` hourly to build history. Z-scores activate after 24 samples (hours).

## Caching

API responses cached in `.cache/` for 10 minutes. Use `--no-cache` to bypass.

## Graceful Degradation

If any API fails, its component is excluded and weight redistributed proportionally. When all data is missing, returns 50 (maximally uncertain).

## Testing

```bash
# Unit tests (no network, mocked APIs)
python3 -m unittest test_bull_score -v

# Include live API integration tests
BULL_LIVE_TESTS=1 python3 -m unittest test_bull_score -v

# Built-in self-test
python3 bull_score.py --self-test
```

72 tests total: 63 unit tests + 9 integration tests.

## Deployment (Mac Mini)

See `deploy/` for LaunchAgent plists and install script.

**Hourly collection** via LaunchAgent:
```bash
# Install on Mac Mini
scp bull_score.py aj@100.71.249.98:/Users/aj/bullscore/
scp deploy/*.plist aj@100.71.249.98:~/Library/LaunchAgents/
ssh aj@100.71.249.98 "launchctl load ~/Library/LaunchAgents/co.hyperday.bullscore.collect.plist"
ssh aj@100.71.249.98 "launchctl load ~/Library/LaunchAgents/co.hyperday.bullscore.deliver.plist"
```

**9am Telegram delivery** via second LaunchAgent that calls `--deliver`.
