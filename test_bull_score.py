#!/usr/bin/env python3
"""
Comprehensive test suite for bull_score.py.

Run unit tests:
    python3 -m unittest test_bull_score -v

Run with live API integration tests:
    BULL_LIVE_TESTS=1 python3 -m unittest test_bull_score -v
"""

import json
import math
import os
import sqlite3
import statistics
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

# Import the module under test
import bull_score as bs


# ─── Unit Tests: Pure Math ───────────────────────────────────────────────────


class TestClamp(unittest.TestCase):
    def test_below_range(self):
        self.assertEqual(bs._clamp(-1.0), 0.0)

    def test_at_lower_bound(self):
        self.assertEqual(bs._clamp(0.0), 0.0)

    def test_in_range(self):
        self.assertEqual(bs._clamp(0.5), 0.5)

    def test_at_upper_bound(self):
        self.assertEqual(bs._clamp(1.0), 1.0)

    def test_above_range(self):
        self.assertEqual(bs._clamp(2.0), 1.0)

    def test_custom_bounds(self):
        self.assertEqual(bs._clamp(5, lo=0, hi=10), 5)
        self.assertEqual(bs._clamp(-5, lo=0, hi=10), 0)
        self.assertEqual(bs._clamp(15, lo=0, hi=10), 10)


class TestNormalCdf(unittest.TestCase):
    def test_zero(self):
        self.assertAlmostEqual(bs._normal_cdf(0), 0.5, places=5)

    def test_positive(self):
        self.assertGreater(bs._normal_cdf(1.0), 0.5)
        self.assertAlmostEqual(bs._normal_cdf(1.0), 0.8413, places=3)

    def test_negative(self):
        self.assertLess(bs._normal_cdf(-1.0), 0.5)
        self.assertAlmostEqual(bs._normal_cdf(-1.0), 0.1587, places=3)

    def test_extreme(self):
        self.assertAlmostEqual(bs._normal_cdf(3.0), 0.9987, places=3)
        self.assertAlmostEqual(bs._normal_cdf(-3.0), 0.0013, places=3)


class TestSigmoid(unittest.TestCase):
    def test_center(self):
        self.assertAlmostEqual(bs._sigmoid(0, center=0, scale=1), 0.5, places=5)

    def test_above_center(self):
        self.assertGreater(bs._sigmoid(1, center=0, scale=1), 0.5)

    def test_below_center(self):
        self.assertLess(bs._sigmoid(-1, center=0, scale=1), 0.5)

    def test_zero_scale(self):
        self.assertAlmostEqual(bs._sigmoid(5, center=0, scale=0), 0.5, places=5)


# ─── Unit Tests: Z-Score Engine ──────────────────────────────────────────────


class TestZscoreToScore(unittest.TestCase):
    def test_insufficient_history(self):
        """Returns None when fewer than ZSCORE_MIN_SAMPLES samples."""
        history = [1.0] * 10
        self.assertIsNone(bs.zscore_to_score(1.0, history))

    def test_zero_stdev(self):
        """Returns 0.5 when all values are identical."""
        history = [5.0] * 30
        self.assertAlmostEqual(bs.zscore_to_score(5.0, history), 0.5, places=3)

    def test_above_mean(self):
        """Value above mean should score > 0.5."""
        history = list(range(30))  # mean=14.5, std=8.8
        score = bs.zscore_to_score(25, history)
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.5)

    def test_below_mean(self):
        """Value below mean should score < 0.5."""
        history = list(range(30))
        score = bs.zscore_to_score(5, history)
        self.assertIsNotNone(score)
        self.assertLess(score, 0.5)

    def test_at_mean(self):
        """Value at mean should score ~0.5."""
        history = list(range(30))
        mu = statistics.mean(history)
        score = bs.zscore_to_score(mu, history)
        self.assertIsNotNone(score)
        self.assertAlmostEqual(score, 0.5, places=2)

    def test_inverted(self):
        """Invert=True flips the score."""
        history = list(range(30))
        normal = bs.zscore_to_score(25, history)
        inverted = bs.zscore_to_score(25, history, invert=True)
        self.assertIsNotNone(normal)
        self.assertIsNotNone(inverted)
        self.assertAlmostEqual(normal + inverted, 1.0, places=2)

    def test_extreme_values_capped(self):
        """Z-scores are capped at +/- 3, so scores stay in [0.0013, 0.9987]."""
        history = [50.0] * 25 + [51.0] * 5  # low stdev
        score = bs.zscore_to_score(1000, history)
        self.assertIsNotNone(score)
        self.assertLessEqual(score, 1.0)
        self.assertGreaterEqual(score, 0.0)

    def test_returns_float(self):
        """Score should always be a float."""
        history = list(range(30))
        score = bs.zscore_to_score(15, history)
        self.assertIsInstance(score, float)


# ─── Unit Tests: Component Scorers ───────────────────────────────────────────


class TestScoreLiquidity(unittest.TestCase):
    def test_missing_data(self):
        self.assertIsNone(bs.score_liquidity({}))
        self.assertIsNone(bs.score_liquidity(None))

    def test_bullish_stablecoins(self):
        """Strong stablecoin growth should score high."""
        data = {"pct_7d": 0.01, "pct_30d": 0.03, "pct_90d": 0.06, "supply_today": 1e11}
        result = bs.score_liquidity(data)
        self.assertIsNotNone(result)
        score, raw = result
        self.assertGreater(score, 0.7)
        self.assertLessEqual(score, 1.0)

    def test_bearish_stablecoins(self):
        """Contracting stablecoins should score low."""
        data = {"pct_7d": -0.01, "pct_30d": -0.03, "pct_90d": -0.05, "supply_today": 1e11}
        result = bs.score_liquidity(data)
        self.assertIsNotNone(result)
        score, raw = result
        self.assertLess(score, 0.3)

    def test_v1_mode(self):
        """v1 uses only 30d with original clamp."""
        data = {"pct_30d": 0.02, "supply_today": 1e11}
        result = bs.score_liquidity(data, mode="v1")
        self.assertIsNotNone(result)
        score, raw = result
        expected = bs._clamp((0.02 + 0.02) / 0.06)
        self.assertAlmostEqual(score, expected, places=3)


class TestScoreTrend(unittest.TestCase):
    def test_missing_data(self):
        self.assertIsNone(bs.score_trend(None, None, None))

    def test_above_ma200(self):
        """Price well above MA200 should score high."""
        result = bs.score_trend(100000, 95000, 80000)
        self.assertIsNotNone(result)
        score, raw = result
        self.assertGreater(score, 0.5)

    def test_below_ma200(self):
        """Price well below MA200 should score low."""
        result = bs.score_trend(60000, 65000, 80000)
        self.assertIsNotNone(result)
        score, raw = result
        self.assertLess(score, 0.3)

    def test_v1_binary(self):
        """v1 uses binary cross."""
        result = bs.score_trend(81000, 82000, 80000, mode="v1")
        self.assertIsNotNone(result)
        score, raw = result
        self.assertEqual(score, 1.0)  # above 200 + golden cross

        result = bs.score_trend(79000, 78000, 80000, mode="v1")
        score, raw = result
        self.assertEqual(score, 0.0)  # below 200 + death cross


class TestScoreActivity(unittest.TestCase):
    def test_missing_all(self):
        self.assertIsNone(bs.score_activity({}, None, None))

    def test_bullish_dex(self):
        """Strong DEX growth should push score up."""
        dex_data = {"change_1d": 15, "change_7d": 25, "change_1m": 30}
        result = bs.score_activity(dex_data, 1.2, 30)
        self.assertIsNotNone(result)
        score, raw = result
        self.assertGreater(score, 0.5)

    def test_bearish_dex(self):
        """Declining DEX should push score down."""
        dex_data = {"change_1d": -15, "change_7d": -25, "change_1m": -35}
        result = bs.score_activity(dex_data, 0.7, 2)
        self.assertIsNotNone(result)
        score, raw = result
        self.assertLess(score, 0.3)


class TestScoreLeverage(unittest.TestCase):
    def test_missing_data(self):
        self.assertIsNone(bs.score_leverage(None, None))

    def test_zero_funding_not_zero(self):
        """Zero funding should score ~0.35, NOT zero (v2 fix)."""
        result = bs.score_leverage(0.0, 0.5)
        self.assertIsNotNone(result)
        score, raw = result
        self.assertGreater(score, 0.25)
        self.assertLess(score, 0.5)

    def test_healthy_bull(self):
        """0.01% funding rate is healthy bull territory."""
        result = bs.score_leverage(0.0001, 0.8)
        self.assertIsNotNone(result)
        score, raw = result
        self.assertGreater(score, 0.5)
        self.assertLess(score, 1.0)

    def test_overleveraged(self):
        """Very high funding rate signals overleveraging."""
        result = bs.score_leverage(0.0005, 0.95)
        self.assertIsNotNone(result)
        score, raw = result
        self.assertLess(score, 0.7)

    def test_deeply_negative(self):
        """Deeply negative funding = bears in control."""
        result = bs.score_leverage(-0.001, 0.1)
        self.assertIsNotNone(result)
        score, raw = result
        self.assertLess(score, 0.15)

    def test_v1_mode(self):
        """v1 uses original linear mapping."""
        result = bs.score_leverage(0.0001, None, mode="v1")
        self.assertIsNotNone(result)
        score, raw = result
        expected = 0.0001 / 0.0002
        self.assertAlmostEqual(score, expected, places=3)


class TestScoreMacro(unittest.TestCase):
    def test_missing_data(self):
        self.assertIsNone(bs.score_macro(None))

    def test_expansion(self):
        """Positive WALCL change should score > 0.5."""
        result = bs.score_macro(0.01)
        self.assertIsNotNone(result)
        score, raw = result
        self.assertGreater(score, 0.5)

    def test_contraction(self):
        """Negative WALCL change should score < 0.5."""
        result = bs.score_macro(-0.01)
        self.assertIsNotNone(result)
        score, raw = result
        self.assertLess(score, 0.5)


# ─── Unit Tests: Weight Redistribution ───────────────────────────────────────


class TestWeightRedistribution(unittest.TestCase):
    @patch('bull_score.fetch_stablecoin_supply', return_value={"pct_7d": 0.01, "pct_30d": 0.02, "pct_90d": 0.03, "supply_today": 1e11})
    @patch('bull_score.fetch_dex_overview', return_value={"change_1d": 5, "change_7d": 10, "change_1m": 15})
    @patch('bull_score.fetch_fees_overview', return_value=(1.1, {"fees_accel": 1.1}))
    @patch('bull_score.fetch_btc_prices', return_value=(90000, 85000, 80000, {}))
    @patch('bull_score.fetch_funding_rates', return_value=(0.0001, 0.7, {}))
    @patch('bull_score.fetch_open_interest', return_value=(50000, {}))
    @patch('bull_score.fetch_mempool_fees', return_value=(15, {}))
    @patch('bull_score.fetch_fred_walcl', return_value=(None, {}))
    def test_no_macro_weights_sum_to_one(self, *mocks):
        """When macro is missing, remaining weights should sum to ~1.0."""
        result = bs.compute_score()
        ew = result["effective_weights"]
        total = sum(ew.values())
        self.assertAlmostEqual(total, 1.0, places=3)
        self.assertEqual(ew["MacroScore"], 0.0)

    @patch('bull_score.fetch_stablecoin_supply', return_value={"pct_7d": 0.01, "pct_30d": 0.02, "pct_90d": 0.03, "supply_today": 1e11})
    @patch('bull_score.fetch_dex_overview', return_value={})
    @patch('bull_score.fetch_fees_overview', return_value=(None, {}))
    @patch('bull_score.fetch_btc_prices', return_value=(None, None, None, {}))
    @patch('bull_score.fetch_funding_rates', return_value=(0.0001, 0.7, {}))
    @patch('bull_score.fetch_open_interest', return_value=(None, {}))
    @patch('bull_score.fetch_mempool_fees', return_value=(None, {}))
    @patch('bull_score.fetch_fred_walcl', return_value=(None, {}))
    def test_multiple_missing_still_sums(self, *mocks):
        """With multiple missing components, available weights still sum to 1.0."""
        result = bs.compute_score()
        ew = result["effective_weights"]
        total = sum(ew.values())
        self.assertAlmostEqual(total, 1.0, places=3)


class TestFinalScoreBounds(unittest.TestCase):
    @patch('bull_score.fetch_stablecoin_supply', return_value={"pct_7d": 0.02, "pct_30d": 0.05, "pct_90d": 0.10, "supply_today": 1e11})
    @patch('bull_score.fetch_dex_overview', return_value={"change_1d": 30, "change_7d": 40, "change_1m": 50})
    @patch('bull_score.fetch_fees_overview', return_value=(1.5, {}))
    @patch('bull_score.fetch_btc_prices', return_value=(120000, 100000, 80000, {}))
    @patch('bull_score.fetch_funding_rates', return_value=(0.0001, 0.9, {}))
    @patch('bull_score.fetch_open_interest', return_value=(80000, {}))
    @patch('bull_score.fetch_mempool_fees', return_value=(40, {}))
    @patch('bull_score.fetch_fred_walcl', return_value=(0.02, {}))
    def test_all_bullish(self, *mocks):
        """All bullish inputs should produce score > 70."""
        result = bs.compute_score()
        self.assertGreaterEqual(result["bull_confidence"], 70)
        self.assertLessEqual(result["bull_confidence"], 100)

    @patch('bull_score.fetch_stablecoin_supply', return_value={"pct_7d": -0.02, "pct_30d": -0.05, "pct_90d": -0.08, "supply_today": 1e11})
    @patch('bull_score.fetch_dex_overview', return_value={"change_1d": -20, "change_7d": -30, "change_1m": -40})
    @patch('bull_score.fetch_fees_overview', return_value=(0.5, {}))
    @patch('bull_score.fetch_btc_prices', return_value=(50000, 55000, 80000, {}))
    @patch('bull_score.fetch_funding_rates', return_value=(-0.001, 0.1, {}))
    @patch('bull_score.fetch_open_interest', return_value=(20000, {}))
    @patch('bull_score.fetch_mempool_fees', return_value=(2, {}))
    @patch('bull_score.fetch_fred_walcl', return_value=(-0.02, {}))
    def test_all_bearish(self, *mocks):
        """All bearish inputs should produce score < 20."""
        result = bs.compute_score()
        self.assertLessEqual(result["bull_confidence"], 20)
        self.assertGreaterEqual(result["bull_confidence"], 0)

    @patch('bull_score.fetch_stablecoin_supply', return_value={})
    @patch('bull_score.fetch_dex_overview', return_value={})
    @patch('bull_score.fetch_fees_overview', return_value=(None, {}))
    @patch('bull_score.fetch_btc_prices', return_value=(None, None, None, {}))
    @patch('bull_score.fetch_funding_rates', return_value=(None, None, {}))
    @patch('bull_score.fetch_open_interest', return_value=(None, {}))
    @patch('bull_score.fetch_mempool_fees', return_value=(None, {}))
    @patch('bull_score.fetch_fred_walcl', return_value=(None, {}))
    def test_all_missing_returns_50(self, *mocks):
        """When all data is missing, score should be 50 (maximally uncertain)."""
        result = bs.compute_score()
        self.assertEqual(result["bull_confidence"], 50)


# ─── Unit Tests: Retry & Network ─────────────────────────────────────────────


class TestFetchJsonRetries(unittest.TestCase):
    @patch('bull_score._cache_read', return_value=None)
    @patch('bull_score._cache_write')
    @patch('bull_score.requests.get')
    def test_retry_on_500(self, mock_get, mock_cw, mock_cr):
        """Retries on 500 and succeeds on 3rd attempt."""
        resp_500 = MagicMock(status_code=500)
        resp_200 = MagicMock(status_code=200)
        resp_200.json.return_value = {"ok": True}
        mock_get.side_effect = [resp_500, resp_500, resp_200]

        result = bs.fetch_json("https://example.com/api", max_retries=3, base_delay=0.01)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(mock_get.call_count, 3)

    @patch('bull_score._cache_read', return_value=None)
    @patch('bull_score._cache_write')
    @patch('bull_score.requests.get')
    def test_retry_on_timeout(self, mock_get, mock_cw, mock_cr):
        """Retries on timeout, succeeds eventually."""
        import requests as req
        mock_get.side_effect = [
            req.exceptions.Timeout("timeout"),
            MagicMock(status_code=200, json=lambda: {"ok": True})
        ]
        result = bs.fetch_json("https://example.com/api", max_retries=3, base_delay=0.01)
        self.assertEqual(result, {"ok": True})

    @patch('bull_score._cache_read', return_value=None)
    @patch('bull_score.requests.get')
    def test_no_retry_on_404(self, mock_get, mock_cr):
        """404 fails immediately without retrying."""
        mock_get.return_value = MagicMock(status_code=404)
        result = bs.fetch_json("https://example.com/missing", max_retries=3, base_delay=0.01)
        self.assertIsNone(result)
        self.assertEqual(mock_get.call_count, 1)

    @patch('bull_score._cache_read', return_value=None)
    @patch('bull_score._cache_write')
    @patch('bull_score.requests.get')
    def test_429_respects_retry_after(self, mock_get, mock_cw, mock_cr):
        """Uses Retry-After header value on 429."""
        resp_429 = MagicMock(status_code=429)
        resp_429.headers = {"Retry-After": "1"}
        resp_200 = MagicMock(status_code=200)
        resp_200.json.return_value = {"ok": True}
        mock_get.side_effect = [resp_429, resp_200]

        result = bs.fetch_json("https://example.com/api", max_retries=3, base_delay=0.01)
        self.assertEqual(result, {"ok": True})

    @patch('bull_score._cache_read', return_value=None)
    @patch('bull_score._cache_write')
    @patch('bull_score.requests.get')
    def test_connection_error_retries(self, mock_get, mock_cw, mock_cr):
        """Connection errors trigger retries."""
        import requests as req
        mock_get.side_effect = [
            req.exceptions.ConnectionError("refused"),
            MagicMock(status_code=200, json=lambda: {"data": 1})
        ]
        result = bs.fetch_json("https://example.com/api", max_retries=3, base_delay=0.01)
        self.assertEqual(result, {"data": 1})
        self.assertEqual(mock_get.call_count, 2)

    def test_cache_hit(self):
        """Cached response returned without HTTP call."""
        with patch('bull_score._cache_read', return_value={"cached": True}):
            with patch('bull_score.requests.get') as mock_get:
                result = bs.fetch_json("https://example.com/cached")
                self.assertEqual(result, {"cached": True})
                mock_get.assert_not_called()


# ─── Unit Tests: SQLite Storage ──────────────────────────────────────────────


class TestSQLiteStorage(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.conn = bs.init_db(self.db_path)

    def tearDown(self):
        self.conn.close()
        try:
            os.unlink(self.db_path)
            os.rmdir(self.tmpdir)
        except OSError:
            pass

    def test_init_creates_table(self):
        """Table should exist after init."""
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='snapshots'"
        )
        self.assertEqual(cursor.fetchone()[0], "snapshots")

    def test_store_and_retrieve(self):
        """Round-trip store -> get_history."""
        bs.store_snapshot(self.conn, "test_source", {"metric_a": 42.0, "metric_b": 99.0})
        history = bs.get_history(self.conn, "test_source", "metric_a", hours=1)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0], 42.0)

    def test_duplicate_insert_ignored(self):
        """UNIQUE constraint prevents duplicate inserts."""
        bs.store_snapshot(self.conn, "dup_test", {"val": 1.0})
        bs.store_snapshot(self.conn, "dup_test", {"val": 2.0})  # same hour -> ignored
        count = self.conn.execute("SELECT COUNT(*) FROM snapshots WHERE source='dup_test'").fetchone()[0]
        self.assertEqual(count, 1)  # only first insert kept

    def test_history_window(self):
        """get_history with hours=24 filters correctly."""
        # Insert data at different timestamps
        now = datetime.now(timezone.utc)
        for i in range(48):
            ts = (now - timedelta(hours=i)).strftime("%Y-%m-%dT%H:00:00Z")
            self.conn.execute(
                "INSERT OR IGNORE INTO snapshots (ts, source, metric, value) VALUES (?, ?, ?, ?)",
                (ts, "window_test", "val", float(i))
            )
        self.conn.commit()

        history = bs.get_history(self.conn, "window_test", "val", hours=24)
        self.assertLessEqual(len(history), 25)  # at most 24h of data
        self.assertGreater(len(history), 0)

    def test_health_check_fresh(self):
        """Health check shows 'healthy' with recent data."""
        bs.store_snapshot(self.conn, "stablecoin", {"pct_30d": 0.01})
        bs.store_snapshot(self.conn, "dex", {"change_1m": 5.0})
        health = bs.health_check(self.conn)
        self.assertEqual(health["status"], "healthy")
        self.assertGreater(health["total_snapshots"], 0)

    def test_health_check_empty_db(self):
        """Empty DB should be 'unhealthy'."""
        health = bs.health_check(self.conn)
        self.assertEqual(health["status"], "unhealthy")

    def test_store_ignores_nan(self):
        """NaN and inf values should not be stored."""
        count = bs.store_snapshot(self.conn, "nan_test", {
            "good": 42.0,
            "nan": float('nan'),
            "inf": float('inf'),
        })
        self.assertEqual(count, 1)  # only the good value


# ─── Unit Tests: Telegram Message Formatting ─────────────────────────────────


class TestFormatTelegramMessage(unittest.TestCase):
    def test_format_bullish(self):
        result = {
            "bull_confidence": 75,
            "scoring": "v3-zscore",
            "component_scores": {
                "LiquidityScore": 0.8,
                "TrendScore": 0.7,
                "ActivityScore": 0.6,
                "LeverageScore": 0.5,
                "MacroScore": 0.9,
            },
            "raw_features": {
                "stable_pct_30d": 2.5,
                "close_to_ma200": 1.15,
                "funding_7d_avg": 0.0001,
            },
            "timestamp_utc": "2026-03-05T09:00:00Z",
        }
        msg = bs.format_telegram_message(result)
        self.assertIn("75/100", msg)
        self.assertIn("Liquidity: 80%", msg)
        self.assertIn("growing", msg)

    def test_format_bearish(self):
        result = {
            "bull_confidence": 15,
            "scoring": "v2-clamp",
            "component_scores": {
                "LiquidityScore": 0.1,
                "TrendScore": 0.05,
                "ActivityScore": 0.08,
                "LeverageScore": 0.2,
                "MacroScore": None,
            },
            "raw_features": {
                "stable_pct_30d": -1.5,
                "close_to_ma200": 0.76,
                "funding_7d_avg": -0.0002,
            },
            "timestamp_utc": "2026-03-05T09:00:00Z",
        }
        msg = bs.format_telegram_message(result)
        self.assertIn("15/100", msg)
        self.assertIn("contracting", msg)
        self.assertIn("below MA200", msg)


# ─── Unit Tests: Collect All ─────────────────────────────────────────────────


class TestCollectAll(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.conn = bs.init_db(self.db_path)

    def tearDown(self):
        self.conn.close()
        try:
            os.unlink(self.db_path)
            os.rmdir(self.tmpdir)
        except OSError:
            pass

    @patch('bull_score.fetch_stablecoin_supply', return_value={"pct_7d": 0.01, "pct_30d": 0.02, "pct_90d": 0.03, "supply_today": 1e11})
    @patch('bull_score.fetch_dex_overview', return_value={"change_1d": 5, "change_7d": 10, "change_1m": 15})
    @patch('bull_score.fetch_fees_overview', return_value=(1.1, {}))
    @patch('bull_score.fetch_btc_prices', return_value=(90000, 85000, 80000, {}))
    @patch('bull_score.fetch_funding_rates', return_value=(0.0001, 0.7, {}))
    @patch('bull_score.fetch_open_interest', return_value=(50000, {}))
    @patch('bull_score.fetch_mempool_fees', return_value=(15, {}))
    @patch('bull_score.fetch_fred_walcl', return_value=(None, {}))
    def test_collect_stores_data(self, *mocks):
        """Collect should store metrics and report sources."""
        summary = bs.collect_all(self.conn)
        self.assertGreater(summary["metrics_stored"], 0)
        self.assertIn("stablecoin", summary["sources_ok"])
        self.assertIn("dex", summary["sources_ok"])
        self.assertIn("fred", summary["sources_failed"])  # no API key


# ─── Integration Tests (Live APIs) ──────────────────────────────────────────

LIVE = os.environ.get("BULL_LIVE_TESTS", "").lower() in ("1", "true", "yes")


@unittest.skipUnless(LIVE, "Set BULL_LIVE_TESTS=1 to run live API tests")
class TestLiveStablecoin(unittest.TestCase):
    def test_live_stablecoin_api(self):
        data = bs.fetch_stablecoin_supply()
        self.assertIsInstance(data, dict)
        self.assertIn("pct_30d", data)
        self.assertIn("pct_7d", data)
        self.assertIn("pct_90d", data)
        self.assertTrue(math.isfinite(data["pct_30d"]))


@unittest.skipUnless(LIVE, "Set BULL_LIVE_TESTS=1 to run live API tests")
class TestLiveDex(unittest.TestCase):
    def test_live_dex_api(self):
        data = bs.fetch_dex_overview()
        self.assertIsInstance(data, dict)
        self.assertIn("change_1m", data)
        # 1d and 7d may or may not be present depending on API
        self.assertTrue(math.isfinite(data["change_1m"]))


@unittest.skipUnless(LIVE, "Set BULL_LIVE_TESTS=1 to run live API tests")
class TestLiveFees(unittest.TestCase):
    def test_live_fees_api(self):
        accel, raw = bs.fetch_fees_overview()
        self.assertIsNotNone(accel)
        self.assertTrue(math.isfinite(accel))
        self.assertIn("fees_accel", raw)


@unittest.skipUnless(LIVE, "Set BULL_LIVE_TESTS=1 to run live API tests")
class TestLiveBtcPrices(unittest.TestCase):
    def test_live_btc_prices(self):
        close, ma50, ma200, raw = bs.fetch_btc_prices()
        self.assertIsNotNone(close)
        self.assertIsNotNone(ma200)
        self.assertGreater(close, 0)
        self.assertGreater(ma200, 0)


@unittest.skipUnless(LIVE, "Set BULL_LIVE_TESTS=1 to run live API tests")
class TestLiveFunding(unittest.TestCase):
    def test_live_funding_rates(self):
        avg, pct_pos, raw = bs.fetch_funding_rates()
        self.assertIsNotNone(avg)
        self.assertTrue(math.isfinite(avg))


@unittest.skipUnless(LIVE, "Set BULL_LIVE_TESTS=1 to run live API tests")
class TestLiveOpenInterest(unittest.TestCase):
    def test_live_open_interest(self):
        oi, raw = bs.fetch_open_interest()
        self.assertIsNotNone(oi)
        self.assertGreater(oi, 0)


@unittest.skipUnless(LIVE, "Set BULL_LIVE_TESTS=1 to run live API tests")
class TestLiveMempool(unittest.TestCase):
    def test_live_mempool(self):
        fee, raw = bs.fetch_mempool_fees()
        self.assertIsNotNone(fee)
        self.assertGreater(fee, 0)


@unittest.skipUnless(LIVE, "Set BULL_LIVE_TESTS=1 to run live API tests")
class TestLiveFullScore(unittest.TestCase):
    def test_live_full_score(self):
        result = bs.compute_score()
        self.assertIn("bull_confidence", result)
        score = result["bull_confidence"]
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        for k, v in result["component_scores"].items():
            if v is not None:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)


@unittest.skipUnless(LIVE, "Set BULL_LIVE_TESTS=1 to run live API tests")
class TestLiveCollectAndReport(unittest.TestCase):
    def test_live_collect_and_report(self):
        """Full cycle: collect -> report with temp DB."""
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "live_test.db")
        try:
            conn = bs.init_db(db_path)
            summary = bs.collect_all(conn)
            self.assertGreater(summary["metrics_stored"], 0)
            self.assertGreater(len(summary["sources_ok"]), 0)

            result = bs.compute_score(conn=conn)
            self.assertIn("bull_confidence", result)
            score = result["bull_confidence"]
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)

            health = bs.health_check(conn)
            self.assertIn(health["status"], ("healthy", "degraded"))
            conn.close()
        finally:
            try:
                os.unlink(db_path)
                os.rmdir(tmpdir)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
