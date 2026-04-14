from __future__ import annotations

import os
from typing import Literal, Protocol, Optional

import pandas as pd
import requests

from trading.models import canonical_symbol


MarketDataSource = Literal["hyperliquid", "binance"]


class MarketDataClient(Protocol):
    def get_historical_ohlcv(
        self,
        symbol: str,
        *,
        start_time: str,
        end_time: str,
        timeframe: str = "1h",
    ) -> pd.DataFrame: ...


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
        return _normalize_hyperliquid_candles(raw)

    def _candle_interval_for_timeframe(self, timeframe: str) -> str:
        normalized = str(timeframe).lower()
        if normalized in {"15m", "1h", "4h", "1d"}:
            return normalized
        raise ValueError(f"unsupported Hyperliquid replay timeframe: {timeframe}")


class BinanceMarketDataClient:
    def __init__(self, *, base_url: Optional[str] = None, timeout: int = 20, limit: int = 1000):
        self.base_url = base_url or os.getenv("BINANCE_BASE_URL") or "https://api.binance.com"
        self.timeout = timeout
        self.limit = limit

    def get_historical_ohlcv(
        self,
        symbol: str,
        *,
        start_time: str,
        end_time: str,
        timeframe: str = "1h",
    ) -> pd.DataFrame:
        interval = self._candle_interval_for_timeframe(timeframe)
        interval_ms = self._interval_ms(interval)
        start_ms = int(pd.Timestamp(start_time, tz="UTC").timestamp() * 1000)
        end_ms = int(pd.Timestamp(end_time, tz="UTC").timestamp() * 1000)
        current_start_ms = start_ms
        rows: list[dict] = []

        while current_start_ms <= end_ms:
            response = requests.get(
                f"{self.base_url}/api/v3/klines",
                params={
                    "symbol": self._binance_symbol(symbol),
                    "interval": interval,
                    "startTime": current_start_ms,
                    "endTime": end_ms,
                    "limit": self.limit,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            raw = response.json()
            if not raw:
                break

            rows.extend(_normalize_binance_klines(raw).to_dict("records"))
            last_open_time = int(raw[-1][0])
            next_start_ms = last_open_time + interval_ms
            if next_start_ms <= current_start_ms:
                break
            current_start_ms = next_start_ms
            if len(raw) < self.limit:
                break

        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame

        start_ts = pd.Timestamp(start_time, tz="UTC")
        end_ts = pd.Timestamp(end_time, tz="UTC")
        frame = frame.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        mask = (frame["Date"] >= start_ts) & (frame["Date"] <= end_ts)
        return frame.loc[mask].reset_index(drop=True)

    def _binance_symbol(self, symbol: str) -> str:
        return f"{canonical_symbol(symbol)}USDT"

    def _candle_interval_for_timeframe(self, timeframe: str) -> str:
        normalized = str(timeframe).lower()
        if normalized in {"15m", "1h", "4h", "1d"}:
            return normalized
        raise ValueError(f"unsupported Binance replay timeframe: {timeframe}")

    def _interval_ms(self, interval: str) -> int:
        mapping = {
            "15m": 15 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }
        return mapping[interval]


def create_market_data_client(source: MarketDataSource, **kwargs) -> MarketDataClient:
    normalized = str(source).lower()
    if normalized == "hyperliquid":
        return HyperliquidMarketDataClient(**kwargs)
    if normalized == "binance":
        return BinanceMarketDataClient(**kwargs)
    raise ValueError(f"unsupported market data source: {source}")


def _normalize_hyperliquid_candles(raw: list[object]) -> pd.DataFrame:
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


def _normalize_binance_klines(raw: list[list[object]]) -> pd.DataFrame:
    rows = []
    for candle in raw:
        if not isinstance(candle, list) or len(candle) < 6:
            continue
        rows.append(
            {
                "Date": pd.to_datetime(candle[0], unit="ms", utc=True),
                "Open": float(candle[1]),
                "High": float(candle[2]),
                "Low": float(candle[3]),
                "Close": float(candle[4]),
                "Volume": float(candle[5]),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("Date").reset_index(drop=True)
