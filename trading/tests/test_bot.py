import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trading.bot import TradingBot
from trading.db import TradingRepository
from trading.models import BotConfig, ExchangeStateSnapshot, ExecutionMode, TradeAction


class FakeExecutor:
    def __init__(self, candles):
        self.candles = candles
        self.intents = []

    def get_historical_ohlcv(self, symbol, *, start_time, end_time, timeframe):
        return self.candles.reset_index().rename(columns={"index": "Date"})

    def get_exchange_state_snapshot(self, symbol):
        return ExchangeStateSnapshot(
            wallet_address="0xabc",
            positions=[],
            open_orders=[],
            mark_prices={"BTC": float(self.candles.iloc[-1]["Close"])},
            fetched_at="2025-01-01T00:00:00+00:00",
        )

    def execute(self, intent):
        self.intents.append(intent)
        raise AssertionError("execute should not be called in paper mode test")


class TradingBotTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo = TradingRepository(Path(self.tempdir.name) / "trading.sqlite3")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_bot_enters_once_per_completed_signal(self):
        candles = pd.DataFrame(
            [
                {"Open": 99.5, "High": 101.0, "Low": 99.0, "Close": 100.0},
                {"Open": 100.0, "High": 102.0, "Low": 99.0, "Close": 101.0},
                {"Open": 101.0, "High": 103.0, "Low": 100.0, "Close": 102.0},
                {"Open": 102.0, "High": 104.0, "Low": 101.0, "Close": 103.0},
                {"Open": 103.0, "High": 105.0, "Low": 102.0, "Close": 104.0},
                {"Open": 104.0, "High": 106.0, "Low": 103.0, "Close": 105.0},
                {"Open": 105.0, "High": 107.0, "Low": 104.0, "Close": 106.0},
                {"Open": 106.0, "High": 107.0, "Low": 104.0, "Close": 105.0},
                {"Open": 105.0, "High": 107.0, "Low": 104.0, "Close": 106.0},
            ],
            index=pd.date_range("2025-01-01", periods=9, freq="15min", tz="UTC"),
        )
        executor = FakeExecutor(candles)
        config = BotConfig(
            strategy_id="trend_pullback",
            symbol="BTC-USD",
            timeframe="15m",
            execution_mode=ExecutionMode.PAPER,
            fixed_notional=100.0,
            leverage=1,
            time_stop_bars=12,
            poll_interval_seconds=5,
            strategy_params={"ema_fast": 3, "ema_trend": 5, "ema_bias": 8, "atr_period": 3},
        )
        bot_run = self.repo.start_bot_run(
            mode=config.execution_mode.value,
            strategy_id=config.strategy_id,
            symbol=config.symbol,
            timeframe=config.timeframe,
            config=config.model_dump(mode="json"),
        )
        bot = TradingBot(
            config=config,
            repository=self.repo,
            executor=executor,
            bot_run_id=bot_run.id,
        )

        first = bot.run_once()
        second = bot.run_once()
        self.assertEqual(first.action, "paper_entry")
        self.assertEqual(second.action, "idle")
        trade_events = self.repo.list_trade_events()
        self.assertTrue(any(event["event_type"] == "paper_entry" for event in trade_events))
