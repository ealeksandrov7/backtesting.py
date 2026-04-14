from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import requests

from trading.models import canonical_symbol


class HyperliquidMarketDataClient:
    def __init__(self, *, base_url: Optional[str] = None, testnet: bool = False, timeout: int = 20):
        self.base_url = base_url or os.getenv("HYPERLIQUID_BASE_URL") or (
            "https://api.hyperliquid-testnet.xyz" if testnet else "https://api.hyperliquid.xyz"
        )
        self.timeout = timeout

    def get_historical_ohlcv(
        self,
        symbol: str,
        *,
        start_time: str,
        end_time: str,
        timeframe: str = "1h",
    ) -> pd.DataFrame:
        interval = self._candle_interval_for_timeframe(timeframe)
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": canonical_symbol(symbol),
                "interval": interval,
                "startTime": int(pd.Timestamp(start_time, tz="UTC").timestamp() * 1000),
                "endTime": int(pd.Timestamp(end_time, tz="UTC").timestamp() * 1000),
            },
        }
        response = requests.post(f"{self.base_url}/info", json=payload, timeout=self.timeout)
        response.raise_for_status()
        raw = response.json()

        rows = []
        for candle in raw:
            if not isinstance(candle, dict):
                continue
            rows.append(
                {
                    "Date": pd.to_datetime(candle.get("t"), unit="ms", utc=True),
                    "Open": float(candle.get("o", 0.0) or 0.0),
                    "High": float(candle.get("h", 0.0) or 0.0),
                    "Low": float(candle.get("l", 0.0) or 0.0),
                    "Close": float(candle.get("c", 0.0) or 0.0),
                    "Volume": float(candle.get("v", 0.0) or 0.0),
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values("Date").reset_index(drop=True)

    def _candle_interval_for_timeframe(self, timeframe: str) -> str:
        normalized = str(timeframe).lower()
        if normalized in {"15m", "1h", "4h", "1d"}:
            return normalized
        raise ValueError(f"unsupported Hyperliquid replay timeframe: {timeframe}")
