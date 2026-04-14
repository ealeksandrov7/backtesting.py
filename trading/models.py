from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


def canonical_symbol(value: str) -> str:
    symbol = value.strip().upper()
    for suffix in ("-USD", "/USD", "USDT", "-PERP"):
        if symbol.endswith(suffix):
            symbol = symbol[: -len(suffix)]
            break
    return symbol


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TradeAction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class ExecutionMode(str, Enum):
    ANALYSIS = "analysis"
    PAPER = "paper"
    LIVE = "live"


class EntryMode(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    LIMIT_ZONE = "LIMIT_ZONE"


class StructuredTradeDecision(BaseModel):
    symbol: str
    timestamp: str
    action: TradeAction
    entry_mode: EntryMode = EntryMode.MARKET
    entry_price: Optional[float] = Field(default=None, gt=0.0)
    entry_zone_low: Optional[float] = Field(default=None, gt=0.0)
    entry_zone_high: Optional[float] = Field(default=None, gt=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    thesis_summary: str
    time_horizon: str
    stop_loss: Optional[float] = Field(default=None, gt=0.0)
    take_profit: Optional[float] = Field(default=None, gt=0.0)
    invalidation: str
    size_hint: Optional[str] = None
    setup_expiry_bars: Optional[int] = Field(default=None, ge=1)
    position_instruction: Optional[str] = None

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        symbol = canonical_symbol(value)
        if not symbol:
            raise ValueError("symbol must not be empty")
        return symbol

    @field_validator("thesis_summary", "time_horizon", "invalidation")
    @classmethod
    def ensure_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("text field must not be empty")
        return text

    @field_validator("size_hint")
    @classmethod
    def normalize_size_hint(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = value.strip()
        return text or None

    @model_validator(mode="after")
    def validate_action_fields(self) -> "StructuredTradeDecision":
        if self.action == TradeAction.FLAT:
            return self

        if self.entry_mode == EntryMode.LIMIT and self.entry_price is None:
            raise ValueError("entry_price is required for LIMIT entries")
        if self.entry_mode == EntryMode.LIMIT_ZONE:
            if self.entry_zone_low is None or self.entry_zone_high is None:
                raise ValueError("entry_zone_low and entry_zone_high are required for LIMIT_ZONE entries")
            if self.entry_zone_low >= self.entry_zone_high:
                raise ValueError("entry_zone_low must be less than entry_zone_high")

        if self.stop_loss is None:
            raise ValueError("stop_loss is required for directional trades")
        if self.take_profit is None:
            raise ValueError("take_profit is required for directional trades")
        return self


class Position(BaseModel):
    symbol: str
    side: TradeAction
    size: float = Field(gt=0.0)
    entry_price: float = Field(gt=0.0)
    stop_loss: Optional[float] = Field(default=None, gt=0.0)
    take_profit: Optional[float] = Field(default=None, gt=0.0)
    opened_at: str
    mode: ExecutionMode

    @field_validator("symbol")
    @classmethod
    def normalize_position_symbol(cls, value: str) -> str:
        return canonical_symbol(value)


class OrderIntent(BaseModel):
    mode: ExecutionMode
    symbol: str
    action: TradeAction
    size: float = Field(ge=0.0)
    reference_price: float = Field(gt=0.0)
    entry_mode: EntryMode = EntryMode.MARKET
    limit_price: Optional[float] = Field(default=None, gt=0.0)
    limit_zone_low: Optional[float] = Field(default=None, gt=0.0)
    limit_zone_high: Optional[float] = Field(default=None, gt=0.0)
    leverage: int = Field(ge=1)
    stop_loss: Optional[float] = Field(default=None, gt=0.0)
    take_profit: Optional[float] = Field(default=None, gt=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    thesis_summary: str
    time_horizon: str
    invalidation: str
    decision_timestamp: str
    rationale: str
    reduce_only: bool = False

    @field_validator("symbol")
    @classmethod
    def normalize_order_symbol(cls, value: str) -> str:
        return canonical_symbol(value)


class OrderStatus(str, Enum):
    PREVIEW = "preview"
    FILLED = "filled"
    REJECTED = "rejected"
    SKIPPED = "skipped"
    ERROR = "error"


class OrderPreview(BaseModel):
    status: OrderStatus
    mode: ExecutionMode
    symbol: str
    action: TradeAction
    message: str
    reference_price: Optional[float] = None
    size: Optional[float] = None
    leverage: Optional[int] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    order_id: Optional[str] = None
    raw_response: Optional[dict] = None


class ExchangeOrder(BaseModel):
    symbol: str
    order_id: str
    side: TradeAction
    size: float = Field(ge=0.0)
    limit_price: Optional[float] = Field(default=None, gt=0.0)
    reduce_only: bool = False
    status: str = "open"

    @field_validator("symbol")
    @classmethod
    def normalize_exchange_order_symbol(cls, value: str) -> str:
        return canonical_symbol(value)


class ExchangeStateSnapshot(BaseModel):
    wallet_address: Optional[str] = None
    equity: Optional[float] = None
    available_balance: Optional[float] = None
    spot_usdc_balance: Optional[float] = None
    mark_prices: dict[str, float] = Field(default_factory=dict)
    positions: list[Position] = Field(default_factory=list)
    open_orders: list[ExchangeOrder] = Field(default_factory=list)
    fetched_at: str


class StrategyParameterSpec(BaseModel):
    name: str
    label: str
    type: str
    default: int | float | str | bool
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    description: str


class StrategyMetadata(BaseModel):
    id: str
    name: str
    description: str
    supported_markets: list[str]
    supported_timeframes: list[str]
    default_params: dict[str, Any]
    param_schema: list[StrategyParameterSpec]


class StrategyDecision(BaseModel):
    strategy_id: str
    symbol: str
    timeframe: str
    timestamp: str
    action: TradeAction
    reference_price: float = Field(gt=0.0)
    stop_loss: float = Field(gt=0.0)
    take_profit: float = Field(gt=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    thesis_summary: str
    invalidation: str
    rationale: str
    time_stop_bars: int = Field(ge=1)


class BacktestRequest(BaseModel):
    strategy_id: str
    symbol: str
    timeframe: str
    start_time: str
    end_time: str
    cash: float = Field(gt=0.0)
    commission: float = Field(ge=0.0)
    spread: float = Field(ge=0.0)
    params: dict[str, Any] = Field(default_factory=dict)


class BacktestResult(BaseModel):
    run_id: int
    metrics: dict[str, Any]
    trades: list[dict[str, Any]]
    plot_path: Optional[str] = None


class BotConfig(BaseModel):
    strategy_id: str
    symbol: str
    timeframe: str
    execution_mode: ExecutionMode
    fixed_notional: float = Field(gt=0.0)
    leverage: int = Field(ge=1)
    time_stop_bars: int = Field(ge=1)
    poll_interval_seconds: int = Field(ge=1)
    strategy_params: dict[str, Any] = Field(default_factory=dict)


class BotRunRecord(BaseModel):
    id: int
    mode: str
    strategy_id: str
    symbol: str
    timeframe: str
    config_json: dict[str, Any]
    started_at: str
    last_heartbeat: Optional[str] = None
    status: str


class TradeLogEvent(BaseModel):
    source: str
    run_id: Optional[int] = None
    bot_run_id: Optional[int] = None
    strategy_id: str
    symbol: str
    timeframe: Optional[str] = None
    event_type: str
    side: Optional[TradeAction] = None
    size: Optional[float] = None
    status: Optional[str] = None
    event_timestamp: str
    entry_timestamp: Optional[str] = None
    exit_timestamp: Optional[str] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    pnl: Optional[float] = None
    fees: Optional[float] = None
    order_id: Optional[str] = None
    notes: Optional[str] = None
    raw_payload_json: Optional[dict[str, Any]] = None
    created_at: str = Field(default_factory=utc_now)
