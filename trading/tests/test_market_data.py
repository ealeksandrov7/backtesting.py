import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from pandas import DatetimeTZDtype

from trading.market_data import (
    BinanceMarketDataClient,
    HyperliquidMarketDataClient,
    create_market_data_client,
)


class BinanceMarketDataClientTest(unittest.TestCase):
    def test_timeframe_mapping_supports_expected_intervals(self):
        client = BinanceMarketDataClient()
        self.assertEqual(client._candle_interval_for_timeframe("15m"), "15m")
        self.assertEqual(client._candle_interval_for_timeframe("1h"), "1h")
        self.assertEqual(client._candle_interval_for_timeframe("4h"), "4h")
        self.assertEqual(client._candle_interval_for_timeframe("1d"), "1d")
        with self.assertRaises(ValueError):
            client._candle_interval_for_timeframe("5m")

    @patch("trading.market_data.requests.get")
    def test_get_historical_ohlcv_paginates_and_normalizes(self, mock_get):
        page_one = MagicMock()
        page_one.raise_for_status.return_value = None
        page_one.json.return_value = [
            [1735689600000, "100.0", "101.0", "99.0", "100.5", "10.0"],
            [1735690500000, "100.5", "102.0", "100.0", "101.5", "12.5"],
        ]
        page_two = MagicMock()
        page_two.raise_for_status.return_value = None
        page_two.json.return_value = [
            [1735691400000, "101.5", "103.0", "101.0", "102.5", "8.0"],
        ]
        mock_get.side_effect = [page_one, page_two]

        client = BinanceMarketDataClient(limit=2)
        frame = client.get_historical_ohlcv(
            "BTC-USD",
            start_time="2025-01-01T00:00:00+00:00",
            end_time="2025-01-01T00:30:00+00:00",
            timeframe="15m",
        )

        self.assertEqual(len(frame), 3)
        self.assertListEqual(list(frame.columns), ["Date", "Open", "High", "Low", "Close", "Volume"])
        self.assertIsInstance(frame["Date"].dtype, DatetimeTZDtype)
        self.assertEqual(str(frame.iloc[0]["Date"]), "2025-01-01 00:00:00+00:00")
        self.assertEqual(float(frame.iloc[-1]["Close"]), 102.5)
        self.assertEqual(float(frame.iloc[-1]["Volume"]), 8.0)
        self.assertEqual(mock_get.call_count, 2)
        first_call_params = mock_get.call_args_list[0].kwargs["params"]
        self.assertEqual(first_call_params["symbol"], "BTCUSDT")
        self.assertEqual(first_call_params["interval"], "15m")


class MarketDataFactoryTest(unittest.TestCase):
    def test_create_market_data_client_supports_known_sources(self):
        self.assertIsInstance(create_market_data_client("hyperliquid"), HyperliquidMarketDataClient)
        self.assertIsInstance(create_market_data_client("binance"), BinanceMarketDataClient)
        with self.assertRaises(ValueError):
            create_market_data_client("yahoo")
