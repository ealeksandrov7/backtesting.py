import unittest

import pandas as pd

from trading.models import TradeAction
from trading.strategies.trend_pullback import (
    TrendPullbackParams,
    compute_trend_pullback_frame,
    latest_trend_pullback_decision,
)


class TrendPullbackStrategyTest(unittest.TestCase):
    def setUp(self):
        self.params = TrendPullbackParams(
            ema_fast=3,
            ema_trend=5,
            ema_bias=8,
            atr_period=3,
            higher_tf_validation=False,
            allow_long=True,
            allow_short=True,
            require_vwap_confirmation=False,
            rsi_trend_threshold=50.0,
            rsi_long_pullback_floor=0.0,
            rsi_long_pullback_ceiling=100.0,
            rsi_short_pullback_floor=0.0,
            rsi_short_pullback_ceiling=100.0,
            rsi_turn_threshold=0.0,
        )

    def test_long_signal_generation(self):
        frame = pd.DataFrame(
            [
                {"Open": 99.5, "High": 101.0, "Low": 99.0, "Close": 100.0},
                {"Open": 100.0, "High": 102.0, "Low": 99.0, "Close": 101.0},
                {"Open": 101.0, "High": 103.0, "Low": 100.0, "Close": 102.0},
                {"Open": 102.0, "High": 104.0, "Low": 101.0, "Close": 103.0},
                {"Open": 103.0, "High": 105.0, "Low": 102.0, "Close": 104.0},
                {"Open": 104.0, "High": 106.0, "Low": 103.0, "Close": 105.0},
                {"Open": 105.0, "High": 108.0, "Low": 104.0, "Close": 106.0},
                {"Open": 106.0, "High": 107.0, "Low": 104.0, "Close": 105.0},
                {"Open": 105.0, "High": 109.0, "Low": 104.0, "Close": 106.0},
            ],
            index=pd.date_range("2025-01-01", periods=9, freq="15min", tz="UTC"),
        )
        enriched = compute_trend_pullback_frame(frame, self.params)
        self.assertEqual(int(enriched.iloc[-1]["signal"]), 1)

        decision = latest_trend_pullback_decision(frame, "BTC-USD", "15m", self.params)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, TradeAction.LONG)
        self.assertGreater(decision.take_profit, decision.reference_price)
        self.assertLess(decision.stop_loss, decision.reference_price)

    def test_short_signal_generation(self):
        frame = pd.DataFrame(
            [
                {"Open": 107.5, "High": 108.5, "Low": 106.0, "Close": 107.0},
                {"Open": 107.0, "High": 108.0, "Low": 105.0, "Close": 106.0},
                {"Open": 106.0, "High": 107.0, "Low": 104.0, "Close": 105.0},
                {"Open": 105.0, "High": 106.0, "Low": 103.0, "Close": 104.0},
                {"Open": 104.0, "High": 105.0, "Low": 102.0, "Close": 103.0},
                {"Open": 103.0, "High": 104.0, "Low": 101.0, "Close": 102.0},
                {"Open": 102.0, "High": 103.0, "Low": 100.0, "Close": 101.0},
                {"Open": 101.0, "High": 103.0, "Low": 100.0, "Close": 102.0},
                {"Open": 102.0, "High": 103.0, "Low": 98.0, "Close": 99.0},
            ],
            index=pd.date_range("2025-01-01", periods=9, freq="15min", tz="UTC"),
        )
        enriched = compute_trend_pullback_frame(frame, self.params)
        self.assertEqual(int(enriched.iloc[-1]["signal"]), -1)

        decision = latest_trend_pullback_decision(frame, "BTC-USD", "15m", self.params)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, TradeAction.SHORT)
        self.assertLess(decision.take_profit, decision.reference_price)
        self.assertGreater(decision.stop_loss, decision.reference_price)
