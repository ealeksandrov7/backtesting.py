import unittest

from trading.strategy_registry import build_default_registry


class StrategyRegistryTest(unittest.TestCase):
    def test_registry_exposes_trend_pullback_metadata(self):
        registry = build_default_registry()
        metadata = registry.get("trend_pullback").metadata
        self.assertEqual(metadata.name, "Trend Pullback")
        self.assertEqual(metadata.supported_markets, ["BTC-USD"])
        self.assertEqual(metadata.supported_timeframes, ["15m"])
        self.assertTrue(any(item.name == "ema_fast" for item in metadata.param_schema))
        self.assertTrue(any(item.name == "higher_tf_validation" for item in metadata.param_schema))
        self.assertTrue(any(item.name == "allow_long" for item in metadata.param_schema))
        self.assertTrue(any(item.name == "allow_short" for item in metadata.param_schema))
        self.assertTrue(any(item.name == "use_4h_confirmation" for item in metadata.param_schema))
        self.assertTrue(metadata.default_params["allow_long"])
        self.assertTrue(metadata.default_params["allow_short"])
        self.assertTrue(metadata.default_params["use_4h_confirmation"])
