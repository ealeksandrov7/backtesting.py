from __future__ import annotations

import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any, Optional

import pandas as pd

try:
    from eth_account import Account
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Account = None  # type: ignore[assignment]

try:
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils.constants import MAINNET_API_URL, TESTNET_API_URL
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Exchange = None  # type: ignore[assignment]
    Info = None  # type: ignore[assignment]
    MAINNET_API_URL = "https://api.hyperliquid.xyz"
    TESTNET_API_URL = "https://api.hyperliquid-testnet.xyz"

from trading.models import (
    EntryMode,
    ExchangeOrder,
    ExchangeStateSnapshot,
    ExecutionMode,
    OrderIntent,
    OrderPreview,
    OrderStatus,
    Position,
    TradeAction,
    canonical_symbol,
)


class HyperliquidExecutionError(RuntimeError):
    """Raised when Hyperliquid configuration or execution fails."""


DEFAULT_MARKET_SLIPPAGE = 0.05
LIVE_SIZE_DECIMALS = 5


class HyperliquidExecutor:
    def __init__(
        self,
        *,
        wallet_address: Optional[str] = None,
        private_key: Optional[str] = None,
        base_url: Optional[str] = None,
        testnet: bool = False,
    ):
        self.testnet = testnet
        self.base_url = base_url or os.getenv("HYPERLIQUID_BASE_URL") or (
            TESTNET_API_URL if testnet else MAINNET_API_URL
        )
        self.wallet_address = wallet_address or os.getenv("HYPERLIQUID_WALLET_ADDRESS")
        self.private_key = private_key or os.getenv("HYPERLIQUID_PRIVATE_KEY")
        self._require_sdk()
        spot_meta = self._safe_spot_meta()
        self.info = Info(base_url=self.base_url, skip_ws=True, spot_meta=spot_meta)
        self.exchange: Optional[Any] = None

        if self.private_key:
            if Account is None:
                raise HyperliquidExecutionError(
                    "eth-account is not installed. Install trading dependencies first."
                )
            wallet = Account.from_key(self.private_key)
            self.exchange = Exchange(
                wallet=wallet,
                base_url=self.base_url,
                account_address=self.wallet_address or wallet.address,
                spot_meta=spot_meta,
            )
            self.wallet_address = self.wallet_address or wallet.address

    def get_mark_price(self, symbol: str) -> float:
        symbol = canonical_symbol(symbol)
        mids = self.info.all_mids()
        if symbol not in mids:
            raise HyperliquidExecutionError(f"no Hyperliquid mid price for {symbol}")
        return float(mids[symbol])

    def get_open_position(self) -> Optional[Position]:
        positions = self.get_open_positions()
        return positions[0] if positions else None

    def get_open_positions(self) -> list[Position]:
        if not self.wallet_address:
            return []

        state = self.info.user_state(self.wallet_address)
        positions: list[Position] = []
        for asset_position in state.get("assetPositions", []):
            position = asset_position.get("position", {})
            size = float(position.get("szi", "0") or "0")
            if size == 0:
                continue

            side = TradeAction.LONG if size > 0 else TradeAction.SHORT
            positions.append(
                Position(
                    symbol=position["coin"],
                    side=side,
                    size=abs(size),
                    entry_price=float(position.get("entryPx") or 0.0),
                    stop_loss=None,
                    take_profit=None,
                    opened_at="",
                    mode=ExecutionMode.LIVE,
                )
            )
        return positions

    def get_open_orders(self, symbol: Optional[str] = None) -> list[ExchangeOrder]:
        if not self.wallet_address:
            return []
        normalized_symbol = canonical_symbol(symbol) if symbol else None
        raw_orders = self.info.open_orders(self.wallet_address) or []
        orders: list[ExchangeOrder] = []
        for order in raw_orders:
            coin = order.get("coin") or order.get("name")
            if not coin:
                continue
            if normalized_symbol and canonical_symbol(str(coin)) != normalized_symbol:
                continue
            is_buy = bool(order.get("side") == "B" or order.get("isBuy"))
            side = TradeAction.LONG if is_buy else TradeAction.SHORT
            oid = order.get("oid") or order.get("order", {}).get("oid")
            orders.append(
                ExchangeOrder(
                    symbol=coin,
                    order_id=str(oid),
                    side=side,
                    size=abs(float(order.get("sz", "0") or order.get("origSz", "0") or "0")),
                    limit_price=float(order.get("limitPx") or order.get("px") or 0.0) or None,
                    reduce_only=bool(order.get("reduceOnly", False)),
                    status=str(order.get("status", "open")),
                )
            )
        return orders

    def get_historical_ohlcv(
        self,
        symbol: str,
        *,
        start_time: str,
        end_time: str,
        timeframe: str = "1h",
    ) -> pd.DataFrame:
        interval = self._candle_interval_for_timeframe(timeframe)
        symbol = canonical_symbol(symbol)
        start_ts = pd.Timestamp(start_time, tz="UTC")
        end_ts = pd.Timestamp(end_time, tz="UTC")
        raw = self.info.candles_snapshot(
            symbol.upper(),
            interval,
            int(start_ts.timestamp() * 1000),
            int(end_ts.timestamp() * 1000),
        )
        if not isinstance(raw, list):
            raise HyperliquidExecutionError("unexpected candleSnapshot response shape")

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

    def cancel_order(self, symbol: str, order_id: str) -> dict:
        if self.exchange is None:
            raise HyperliquidExecutionError("live cancellation requires HYPERLIQUID_PRIVATE_KEY")
        return self.exchange.cancel(canonical_symbol(symbol), int(order_id))

    def cancel_orders_for_symbol(self, symbol: str) -> list[dict]:
        results = []
        for order in self.get_open_orders(symbol):
            results.append(self.cancel_order(order.symbol, order.order_id))
        return results

    def get_exchange_state_snapshot(self, symbol: Optional[str] = None) -> ExchangeStateSnapshot:
        if not self.wallet_address:
            raise HyperliquidExecutionError("exchange sync requires wallet address")
        user_state = self.info.user_state(self.wallet_address)
        spot_user_state = self.info.spot_user_state(self.wallet_address)
        mids = self.info.all_mids()
        positions = self.get_open_positions()
        normalized_symbol = canonical_symbol(symbol) if symbol else None
        if symbol:
            positions = [
                position
                for position in positions
                if canonical_symbol(position.symbol) == normalized_symbol
            ]
        open_orders = self.get_open_orders(symbol)
        margin_summary = (
            user_state.get("marginSummary")
            or user_state.get("crossMarginSummary")
            or {}
        )
        return ExchangeStateSnapshot(
            wallet_address=self.wallet_address,
            equity=self._to_optional_float(
                user_state.get("marginSummary", {}).get("accountValue")
                if isinstance(user_state.get("marginSummary"), dict)
                else None
            )
            or self._to_optional_float(margin_summary.get("accountValue"))
            or self._to_optional_float(user_state.get("accountValue")),
            available_balance=self._to_optional_float(user_state.get("withdrawable"))
            or self._to_optional_float(margin_summary.get("withdrawable"))
            or self._to_optional_float(user_state.get("availableBalance")),
            spot_usdc_balance=self._extract_spot_usdc_balance(spot_user_state),
            mark_prices={k: float(v) for k, v in mids.items()},
            positions=positions,
            open_orders=open_orders,
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )

    def execute(self, intent: OrderIntent) -> OrderPreview:
        if self.exchange is None:
            raise HyperliquidExecutionError(
                "live execution requires HYPERLIQUID_PRIVATE_KEY"
            )

        normalized_intent = intent.model_copy(update={"size": self._normalize_size(intent.symbol, intent.size)})
        self.exchange.update_leverage(intent.leverage, intent.symbol, is_cross=True)

        if normalized_intent.action == TradeAction.FLAT:
            raw = self.exchange.market_close(normalized_intent.symbol, sz=normalized_intent.size)
            errors = self._extract_order_errors(raw)
            if errors:
                return OrderPreview(
                    status=OrderStatus.REJECTED,
                    mode=ExecutionMode.LIVE,
                    symbol=normalized_intent.symbol,
                    action=normalized_intent.action,
                    message=f"Live market close rejected: {errors[0]}",
                    reference_price=normalized_intent.reference_price,
                    size=normalized_intent.size,
                    leverage=normalized_intent.leverage,
                    raw_response=raw,
                )
            return OrderPreview(
                status=OrderStatus.FILLED,
                mode=ExecutionMode.LIVE,
                symbol=normalized_intent.symbol,
                action=normalized_intent.action,
                message="Submitted live market close.",
                reference_price=normalized_intent.reference_price,
                size=normalized_intent.size,
                leverage=normalized_intent.leverage,
                raw_response=raw,
            )

        if normalized_intent.stop_loss is not None and normalized_intent.take_profit is not None:
            raw = self.exchange.bulk_orders(
                self._build_bracket_order_requests(normalized_intent),
                grouping="normalTpsl",
            )
            errors = self._extract_order_errors(raw)
            if errors:
                return OrderPreview(
                    status=OrderStatus.REJECTED,
                    mode=ExecutionMode.LIVE,
                    symbol=normalized_intent.symbol,
                    action=normalized_intent.action,
                    message=f"Live bracket order rejected: {errors[0]}",
                    reference_price=self._resolve_entry_reference_price(normalized_intent),
                    size=normalized_intent.size,
                    leverage=normalized_intent.leverage,
                    stop_loss=normalized_intent.stop_loss,
                    take_profit=normalized_intent.take_profit,
                    raw_response=raw,
                )
            return OrderPreview(
                status=OrderStatus.PREVIEW,
                mode=ExecutionMode.LIVE,
                symbol=normalized_intent.symbol,
                action=normalized_intent.action,
                message="Submitted live entry with native TP/SL bracket.",
                reference_price=self._resolve_entry_reference_price(normalized_intent),
                size=normalized_intent.size,
                leverage=normalized_intent.leverage,
                stop_loss=normalized_intent.stop_loss,
                take_profit=normalized_intent.take_profit,
                raw_response=raw,
                order_id=self._extract_order_id(raw, index=0),
            )

        if normalized_intent.entry_mode == EntryMode.MARKET:
            raw = self.exchange.market_open(
                normalized_intent.symbol,
                is_buy=normalized_intent.action == TradeAction.LONG,
                sz=normalized_intent.size,
            )
            message = "Submitted live market order."
            status = OrderStatus.FILLED
        else:
            limit_price = self._resolve_limit_price(normalized_intent)
            raw = self.exchange.order(
                normalized_intent.symbol,
                is_buy=normalized_intent.action == TradeAction.LONG,
                sz=normalized_intent.size,
                limit_px=limit_price,
                order_type={"limit": {"tif": "Gtc"}},
                reduce_only=False,
            )
            message = "Submitted live resting limit order."
            status = OrderStatus.PREVIEW
        errors = self._extract_order_errors(raw)
        if errors:
            return OrderPreview(
                status=OrderStatus.REJECTED,
                mode=ExecutionMode.LIVE,
                symbol=normalized_intent.symbol,
                action=normalized_intent.action,
                message=f"Live order rejected: {errors[0]}",
                reference_price=self._resolve_entry_reference_price(normalized_intent),
                size=normalized_intent.size,
                leverage=normalized_intent.leverage,
                stop_loss=normalized_intent.stop_loss,
                take_profit=normalized_intent.take_profit,
                raw_response=raw,
            )
        return OrderPreview(
            status=status,
            mode=ExecutionMode.LIVE,
            symbol=normalized_intent.symbol,
            action=normalized_intent.action,
            message=message,
            reference_price=self._resolve_limit_price(normalized_intent)
            if normalized_intent.entry_mode != EntryMode.MARKET
            else normalized_intent.reference_price,
            size=normalized_intent.size,
            leverage=normalized_intent.leverage,
            stop_loss=normalized_intent.stop_loss,
            take_profit=normalized_intent.take_profit,
            raw_response=raw,
            order_id=self._extract_order_id(raw),
        )

    def _build_bracket_order_requests(self, intent: OrderIntent) -> list[dict]:
        is_buy = intent.action == TradeAction.LONG
        entry_price = self._resolve_parent_limit_price(intent)
        return [
            {
                "coin": intent.symbol,
                "is_buy": is_buy,
                "sz": intent.size,
                "limit_px": entry_price,
                "order_type": self._parent_order_type(intent),
                "reduce_only": False,
            },
            {
                "coin": intent.symbol,
                "is_buy": not is_buy,
                "sz": intent.size,
                "limit_px": intent.take_profit,
                "order_type": {
                    "trigger": {
                        "isMarket": True,
                        "triggerPx": intent.take_profit,
                        "tpsl": "tp",
                    }
                },
                "reduce_only": True,
            },
            {
                "coin": intent.symbol,
                "is_buy": not is_buy,
                "sz": intent.size,
                "limit_px": intent.stop_loss,
                "order_type": {
                    "trigger": {
                        "isMarket": True,
                        "triggerPx": intent.stop_loss,
                        "tpsl": "sl",
                    }
                },
                "reduce_only": True,
            },
        ]

    def _parent_order_type(self, intent: OrderIntent) -> dict:
        if intent.entry_mode == EntryMode.MARKET:
            return {"limit": {"tif": "Ioc"}}
        return {"limit": {"tif": "Gtc"}}

    def _resolve_parent_limit_price(self, intent: OrderIntent) -> float:
        if intent.entry_mode == EntryMode.MARKET:
            return self.exchange._slippage_price(  # type: ignore[attr-defined]
                intent.symbol,
                intent.action == TradeAction.LONG,
                DEFAULT_MARKET_SLIPPAGE,
                px=intent.reference_price,
            )
        return self._resolve_limit_price(intent)

    def _resolve_entry_reference_price(self, intent: OrderIntent) -> float:
        if intent.entry_mode == EntryMode.MARKET:
            return intent.reference_price
        return self._resolve_limit_price(intent)

    def _resolve_limit_price(self, intent: OrderIntent) -> float:
        if intent.limit_price is not None:
            return intent.limit_price
        if intent.limit_zone_low is not None and intent.limit_zone_high is not None:
            if intent.action == TradeAction.SHORT:
                return intent.limit_zone_low
            return intent.limit_zone_high
        return intent.reference_price

    def _extract_order_id(self, raw: dict | None, index: int = 0) -> Optional[str]:
        order_ids = self._extract_order_ids(raw)
        if index >= len(order_ids):
            return None
        return order_ids[index]

    def _extract_order_ids(self, raw: dict | None) -> list[str]:
        if not isinstance(raw, dict):
            return []
        statuses = raw.get("response", {}).get("data", {}).get("statuses")
        if not isinstance(statuses, list):
            return []
        order_ids: list[str] = []
        for status in statuses:
            if not isinstance(status, dict):
                continue
            resting = status.get("resting")
            if isinstance(resting, dict) and resting.get("oid") is not None:
                order_ids.append(str(resting["oid"]))
                continue
            filled = status.get("filled")
            if isinstance(filled, dict) and filled.get("oid") is not None:
                order_ids.append(str(filled["oid"]))
        return order_ids

    def _extract_order_errors(self, raw: dict | None) -> list[str]:
        if not isinstance(raw, dict):
            return []
        statuses = raw.get("response", {}).get("data", {}).get("statuses")
        if not isinstance(statuses, list):
            return []
        errors: list[str] = []
        for status in statuses:
            if isinstance(status, dict) and status.get("error"):
                errors.append(str(status["error"]))
        return errors

    def _normalize_size(self, symbol: str, size: float) -> float:
        decimals = LIVE_SIZE_DECIMALS
        quantized = Decimal(str(size)).quantize(Decimal(10) ** -decimals, rounding=ROUND_DOWN)
        normalized = float(quantized)
        if normalized <= 0:
            raise HyperliquidExecutionError(
                f"size {size} rounds to zero for {symbol} at {decimals} size decimals"
            )
        return normalized

    def _size_decimals(self, symbol: str) -> Optional[int]:
        return LIVE_SIZE_DECIMALS

    def _metadata_size_decimals(self, symbol: str) -> Optional[int]:
        name_to_asset = getattr(self.info, "name_to_asset", None)
        asset_to_sz_decimals = getattr(self.info, "asset_to_sz_decimals", None)
        if not isinstance(name_to_asset, dict) or not isinstance(asset_to_sz_decimals, dict):
            return None
        asset = name_to_asset.get(symbol)
        if asset is None:
            return None
        decimals = asset_to_sz_decimals.get(asset)
        if decimals is None:
            return None
        try:
            return int(decimals)
        except (TypeError, ValueError):
            return None

    def _safe_spot_meta(self) -> dict:
        # The upstream SDK eagerly assumes a populated spot universe during Info()/Exchange()
        # construction. Hyperliquid testnet can return incomplete spot metadata, which breaks
        # perp-only workflows before any actual perp API call is made. We only trade perps here,
        # so an empty spot universe is the safest initialization shape.
        return {"universe": [], "tokens": []}

    def _to_optional_float(self, value) -> Optional[float]:
        try:
            if value in (None, "", "0", "0.0", 0, 0.0):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _extract_spot_usdc_balance(self, spot_user_state: dict | None) -> Optional[float]:
        if not isinstance(spot_user_state, dict):
            return None
        candidates = []
        balances = spot_user_state.get("balances") or spot_user_state.get("tokenBalances") or []
        if isinstance(balances, list):
            candidates.extend(balances)
        for balance in candidates:
            if not isinstance(balance, dict):
                continue
            coin = str(balance.get("coin") or balance.get("token") or balance.get("name") or "").upper()
            if coin != "USDC":
                continue
            for key in ("total", "balance", "hold", "free"):
                amount = self._to_optional_float(balance.get(key))
                if amount is not None:
                    return amount
        return None

    def _candle_interval_for_timeframe(self, timeframe: str) -> str:
        normalized = str(timeframe).lower()
        if normalized == "15m":
            return "15m"
        if normalized == "1h":
            return "1h"
        if normalized == "4h":
            return "4h"
        if normalized == "1d":
            return "1d"
        raise HyperliquidExecutionError(f"unsupported Hyperliquid replay timeframe: {timeframe}")

    def _require_sdk(self) -> None:
        if Info is None or Exchange is None:
            raise HyperliquidExecutionError(
                "Hyperliquid SDK is not installed. Install the trading dependencies first."
            )
