from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from trading.db import TradingRepository
from trading.models import BacktestRequest, BacktestResult, TradeLogEvent, TradeAction
from trading.strategy_registry import StrategyRegistry


class BacktestService:
    def __init__(
        self,
        *,
        repository: TradingRepository,
        registry: StrategyRegistry,
        market_data,
        artifacts_dir: str | Path,
    ):
        self.repository = repository
        self.registry = registry
        self.market_data = market_data
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def run_backtest(self, request: BacktestRequest) -> BacktestResult:
        definition = self.registry.get(request.strategy_id)
        run_id = self.repository.create_backtest_run(
            strategy_id=request.strategy_id,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_time=request.start_time,
            end_time=request.end_time,
            cash=request.cash,
            commission=request.commission,
            spread=request.spread,
            params=request.params,
        )
        try:
            stats, plot_path = self._execute_backtest(run_id, request, definition)
            metrics = self._serialize_metrics(stats)
            trades = self._serialize_trades(stats["_trades"])
            self.repository.complete_backtest_run(run_id, metrics=metrics, plot_path=str(plot_path) if plot_path else None)
            self._log_trades(run_id, request, trades)
            return BacktestResult(
                run_id=run_id,
                metrics=metrics,
                trades=trades,
                plot_path=str(plot_path) if plot_path else None,
            )
        except Exception as exc:
            self.repository.fail_backtest_run(run_id, error=str(exc))
            raise

    def _execute_backtest(self, run_id: int, request: BacktestRequest, definition) -> tuple[pd.Series, Path | None]:
        try:
            from backtesting.lib import FractionalBacktest
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Running backtests requires the local backtesting.py dependencies, including bokeh."
            ) from exc

        data = self.market_data.get_historical_ohlcv(
            request.symbol,
            start_time=request.start_time,
            end_time=request.end_time,
            timeframe=request.timeframe,
        )
        if data.empty:
            raise RuntimeError("no OHLCV data returned for the requested backtest range")
        frame = data.copy()
        if "Date" in frame.columns:
            frame = frame.set_index("Date")
        frame.index = pd.DatetimeIndex(frame.index)
        frame = frame.sort_index()

        strategy_cls = definition.build_backtesting_strategy()
        backtest = FractionalBacktest(
            frame,
            strategy_cls,
            cash=request.cash,
            commission=request.commission,
            spread=request.spread,
            exclusive_orders=True,
            finalize_trades=True,
        )
        stats = backtest.run(**request.params)

        plot_path = self.artifacts_dir / f"backtest-run-{run_id}.html"
        try:
            backtest.plot(results=stats, filename=str(plot_path), open_browser=False)
        except Exception:
            plot_path = None
        return stats, plot_path

    def _serialize_metrics(self, stats: pd.Series) -> dict[str, Any]:
        metrics = {}
        for key, value in stats.items():
            if str(key).startswith("_"):
                continue
            metrics[str(key)] = self._serialize_value(value)
        return metrics

    def _serialize_trades(self, trades_df: pd.DataFrame) -> list[dict[str, Any]]:
        if trades_df.empty:
            return []
        trades = []
        normalized = trades_df.where(pd.notnull(trades_df), None)
        for _, row in normalized.iterrows():
            serialized_row = {}
            for key, value in row.to_dict().items():
                serialized_row[key] = self._serialize_value(value)
            trades.append(serialized_row)
        return trades

    def _serialize_value(self, value: Any) -> Any:
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, pd.Timedelta):
            return str(value)
        if isinstance(value, float):
            return None if math.isnan(value) else float(value)
        if isinstance(value, (int, str, bool)) or value is None:
            return value
        return str(value)

    def _log_trades(self, run_id: int, request: BacktestRequest, trades: list[dict[str, Any]]) -> None:
        for trade in trades:
            is_long = bool(trade.get("Size", 0) > 0)
            self.repository.log_trade_event(
                TradeLogEvent(
                    source="backtest",
                    run_id=run_id,
                    strategy_id=request.strategy_id,
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    event_type="backtest_trade",
                    side=TradeAction.LONG if is_long else TradeAction.SHORT,
                    size=abs(float(trade.get("Size") or 0.0)),
                    status="closed",
                    event_timestamp=str(trade.get("EntryTime") or trade.get("ExitTime") or request.start_time),
                    entry_timestamp=str(trade.get("EntryTime") or request.start_time),
                    exit_timestamp=str(trade.get("ExitTime") or request.end_time),
                    entry_price=float(trade.get("EntryPrice") or 0.0),
                    exit_price=float(trade.get("ExitPrice") or 0.0),
                    stop_loss=float(trade.get("SL")) if trade.get("SL") is not None else None,
                    take_profit=float(trade.get("TP")) if trade.get("TP") is not None else None,
                    pnl=float(trade.get("PnL") or 0.0),
                    raw_payload_json=trade,
                )
            )
