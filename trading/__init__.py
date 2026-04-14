"""Trading app extensions built on top of backtesting.py."""

from trading.db import TradingRepository
from trading.models import (
    BacktestRequest,
    BacktestResult,
    BotConfig,
    BotRunRecord,
    StrategyDecision,
    StrategyMetadata,
    TradeLogEvent,
)
from trading.services import BacktestService
from trading.strategy_registry import StrategyRegistry, build_default_registry

__all__ = [
    "BacktestRequest",
    "BacktestResult",
    "BotConfig",
    "BotRunRecord",
    "StrategyDecision",
    "StrategyMetadata",
    "TradeLogEvent",
    "TradingRepository",
    "BacktestService",
    "StrategyRegistry",
    "build_default_registry",
]
