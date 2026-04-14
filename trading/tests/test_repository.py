import tempfile
import unittest
from pathlib import Path

from trading.db import TradingRepository
from trading.models import TradeAction, TradeLogEvent


class TradingRepositoryTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo = TradingRepository(Path(self.tempdir.name) / "trading.sqlite3")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_roundtrip_backtest_run_and_trade_log(self):
        run_id = self.repo.create_backtest_run(
            strategy_id="trend_pullback",
            symbol="BTC-USD",
            timeframe="15m",
            start_time="2025-01-01T00:00:00+00:00",
            end_time="2025-01-02T00:00:00+00:00",
            cash=10000.0,
            commission=0.0005,
            spread=0.0,
            params={"ema_fast": 20},
        )
        self.repo.complete_backtest_run(run_id, metrics={"Return [%]": 10.0}, plot_path="artifacts/run.html")
        self.repo.log_trade_event(
            TradeLogEvent(
                source="backtest",
                run_id=run_id,
                strategy_id="trend_pullback",
                symbol="BTC-USD",
                timeframe="15m",
                event_type="backtest_trade",
                side=TradeAction.LONG,
                size=0.1,
                status="closed",
                event_timestamp="2025-01-01T01:00:00+00:00",
                entry_timestamp="2025-01-01T01:00:00+00:00",
                exit_timestamp="2025-01-01T02:00:00+00:00",
                entry_price=100.0,
                exit_price=110.0,
                pnl=1.0,
                raw_payload_json={"foo": "bar"},
            )
        )

        runs = self.repo.list_backtest_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["metrics_json"]["Return [%]"], 10.0)

        trades = self.repo.list_trade_events()
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["raw_payload_json"]["foo"], "bar")
