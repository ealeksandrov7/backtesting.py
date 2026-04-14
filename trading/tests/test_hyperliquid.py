import unittest

from trading.hyperliquid import HyperliquidExecutor


class HyperliquidExecutorTest(unittest.TestCase):
    def test_timeframe_mapping_supports_15m(self):
        executor = object.__new__(HyperliquidExecutor)
        self.assertEqual(executor._candle_interval_for_timeframe("15m"), "15m")
        self.assertEqual(executor._candle_interval_for_timeframe("1h"), "1h")
