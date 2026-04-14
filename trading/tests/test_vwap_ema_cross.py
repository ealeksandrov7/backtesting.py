import unittest

import pandas as pd

from trading.models import TradeAction
from trading.strategies.vwap_ema_cross import (
    VwapEmaCrossParams,
    compute_vwap_ema_cross_frame,
    latest_vwap_ema_cross_decision,
)


class VwapEmaCrossStrategyTest(unittest.TestCase):
    def setUp(self):
        self.params = VwapEmaCrossParams(
            ema_fast=3,
            ema_slow=5,
            atr_period=3,
            higher_tf_validation=False,
            allow_long=True,
            allow_short=True,
            rsi_threshold=0.0,
            break_even_enabled=False,
        )

    def test_long_signal_generation(self):
        frame = pd.DataFrame(
            [
                {"Open": 101.0, "High": 101.5, "Low": 99.0, "Close": 100.0, "Volume": 10.0},
                {"Open": 100.0, "High": 100.5, "Low": 98.0, "Close": 99.0, "Volume": 10.0},
                {"Open": 99.0, "High": 99.5, "Low": 97.0, "Close": 98.0, "Volume": 10.0},
                {"Open": 98.0, "High": 100.0, "Low": 97.5, "Close": 99.5, "Volume": 10.0},
                {"Open": 99.5, "High": 102.0, "Low": 99.0, "Close": 101.5, "Volume": 10.0},
            ],
            index=pd.date_range("2025-01-01", periods=5, freq="15min", tz="UTC"),
        )

        enriched = compute_vwap_ema_cross_frame(frame, self.params)
        self.assertEqual(int(enriched.iloc[-1]["signal"]), 1)

        decision = latest_vwap_ema_cross_decision(frame, "BTC-USD", "15m", self.params)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, TradeAction.LONG)
        self.assertGreater(decision.take_profit, decision.reference_price)
        self.assertLess(decision.stop_loss, decision.reference_price)

    def test_short_signal_generation(self):
        frame = pd.DataFrame(
            [
                {"Open": 99.0, "High": 101.0, "Low": 98.5, "Close": 100.0, "Volume": 10.0},
                {"Open": 100.0, "High": 102.0, "Low": 99.5, "Close": 101.0, "Volume": 10.0},
                {"Open": 101.0, "High": 103.0, "Low": 100.5, "Close": 102.0, "Volume": 10.0},
                {"Open": 102.0, "High": 102.5, "Low": 99.5, "Close": 100.5, "Volume": 10.0},
                {"Open": 100.5, "High": 100.8, "Low": 97.0, "Close": 98.0, "Volume": 10.0},
            ],
            index=pd.date_range("2025-01-01", periods=5, freq="15min", tz="UTC"),
        )

        enriched = compute_vwap_ema_cross_frame(frame, self.params)
        self.assertEqual(int(enriched.iloc[-1]["signal"]), -1)

        decision = latest_vwap_ema_cross_decision(frame, "BTC-USD", "15m", self.params)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, TradeAction.SHORT)
        self.assertLess(decision.take_profit, decision.reference_price)
        self.assertGreater(decision.stop_loss, decision.reference_price)
