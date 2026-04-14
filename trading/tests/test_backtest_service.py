import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trading.db import TradingRepository
from trading.models import BacktestRequest
from trading.services import BacktestService
from trading.strategy_registry import build_default_registry


class FakeMarketData:
    def get_historical_ohlcv(self, symbol, *, start_time, end_time, timeframe):
        frame = pd.DataFrame(
            [
                {"Date": "2025-01-01T00:00:00+00:00", "Open": 99.5, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 10.0},
                {"Date": "2025-01-01T00:15:00+00:00", "Open": 100.0, "High": 102.0, "Low": 99.0, "Close": 101.0, "Volume": 10.0},
                {"Date": "2025-01-01T00:30:00+00:00", "Open": 101.0, "High": 103.0, "Low": 100.0, "Close": 102.0, "Volume": 10.0},
                {"Date": "2025-01-01T00:45:00+00:00", "Open": 102.0, "High": 104.0, "Low": 101.0, "Close": 103.0, "Volume": 10.0},
                {"Date": "2025-01-01T01:00:00+00:00", "Open": 103.0, "High": 105.0, "Low": 102.0, "Close": 104.0, "Volume": 10.0},
                {"Date": "2025-01-01T01:15:00+00:00", "Open": 104.0, "High": 106.0, "Low": 103.0, "Close": 105.0, "Volume": 10.0},
                {"Date": "2025-01-01T01:30:00+00:00", "Open": 105.0, "High": 107.0, "Low": 104.0, "Close": 106.0, "Volume": 10.0},
                {"Date": "2025-01-01T01:45:00+00:00", "Open": 106.0, "High": 107.0, "Low": 104.0, "Close": 105.0, "Volume": 10.0},
                {"Date": "2025-01-01T02:00:00+00:00", "Open": 105.0, "High": 107.0, "Low": 104.0, "Close": 106.0, "Volume": 10.0},
            ]
        )
        frame["Date"] = pd.to_datetime(frame["Date"], utc=True)
        return frame


@unittest.skipUnless(importlib.util.find_spec("bokeh"), "bokeh not installed in local environment")
class BacktestServiceTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.repository = TradingRepository(Path(self.tempdir.name) / "trading.sqlite3")
        self.service = BacktestService(
            repository=self.repository,
            registry=build_default_registry(),
            market_data=FakeMarketData(),
            artifacts_dir=Path(self.tempdir.name) / "artifacts",
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_run_backtest_persists_run(self):
        result = self.service.run_backtest(
            BacktestRequest(
                strategy_id="trend_pullback",
                symbol="BTC-USD",
                timeframe="15m",
                start_time="2025-01-01T00:00:00+00:00",
                end_time="2025-01-01T03:00:00+00:00",
                cash=10000.0,
                commission=0.0005,
                spread=0.0,
                params={
                    "ema_fast": 3,
                    "ema_trend": 5,
                    "ema_bias": 8,
                    "atr_period": 3,
                    "higher_tf_validation": False,
                    "allow_long": True,
                    "allow_short": True,
                },
            )
        )
        self.assertGreater(result.run_id, 0)
        self.assertIn("Return [%]", result.metrics)
        self.assertTrue(self.repository.list_backtest_runs())

    def test_serialize_trades_normalizes_timedelta_values(self):
        trades = pd.DataFrame(
            [
                {
                    "EntryTime": pd.Timestamp("2025-01-01T00:00:00+00:00"),
                    "ExitTime": pd.Timestamp("2025-01-01T03:00:00+00:00"),
                    "Duration": pd.Timedelta(hours=3),
                    "PnL": 12.5,
                }
            ]
        )

        serialized = self.service._serialize_trades(trades)

        self.assertEqual(serialized[0]["EntryTime"], "2025-01-01T00:00:00+00:00")
        self.assertEqual(serialized[0]["ExitTime"], "2025-01-01T03:00:00+00:00")
        self.assertEqual(serialized[0]["Duration"], "0 days 03:00:00")
