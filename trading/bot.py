from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Optional

import pandas as pd

from trading.db import TradingRepository, utc_now
from trading.hyperliquid import HyperliquidExecutionError, HyperliquidExecutor
from trading.models import (
    BotConfig,
    EntryMode,
    ExchangeStateSnapshot,
    ExecutionMode,
    OrderIntent,
    OrderStatus,
    TradeAction,
    TradeLogEvent,
)
from trading.strategy_registry import build_default_registry


def _floor_time(timestamp: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    if timeframe != "15m":
        raise ValueError(f"unsupported timeframe for floor: {timeframe}")
    return timestamp.floor("15min")


def _lookback_start(end_time: datetime, timeframe: str) -> str:
    if timeframe == "15m":
        return (end_time - timedelta(days=10)).isoformat()
    if timeframe == "1h":
        return (end_time - timedelta(days=30)).isoformat()
    raise ValueError(f"unsupported bot timeframe: {timeframe}")


@dataclass
class BotIterationResult:
    action: str
    detail: str


class TradingBot:
    def __init__(
        self,
        *,
        config: BotConfig,
        repository: TradingRepository,
        executor,
        registry=None,
        bot_run_id: Optional[int] = None,
    ):
        self.config = config
        self.repository = repository
        self.executor = executor
        self.registry = registry or build_default_registry()
        self.definition = self.registry.get(config.strategy_id)
        self.bot_run_id = bot_run_id

    def run_once(self) -> BotIterationResult:
        if self.bot_run_id is not None:
            self.repository.heartbeat_bot_run(self.bot_run_id)

        completed = self._load_completed_candles()
        if completed.empty:
            return BotIterationResult(action="idle", detail="No completed candles available.")

        if self.config.execution_mode == ExecutionMode.PAPER:
            close_result = self._sync_paper_position(completed)
            if close_result is not None:
                return close_result
            has_open_position = self.repository.get_open_trade(source="paper", symbol=self.config.symbol) is not None
            has_open_orders = False
        else:
            snapshot = self.executor.get_exchange_state_snapshot(self.config.symbol)
            close_result = self._sync_live_position(snapshot, completed)
            if close_result is not None:
                return close_result
            has_open_position = bool(snapshot.positions)
            has_open_orders = bool(snapshot.open_orders)

        decision = self.definition.latest_decision_builder(
            completed,
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            params=self.config.strategy_params,
        )
        if decision is None:
            return BotIterationResult(action="idle", detail="No signal on the latest completed bar.")

        latest_logged_signal = (
            self.repository.latest_signal_timestamp(bot_run_id=self.bot_run_id)
            if self.bot_run_id is not None
            else None
        )
        if latest_logged_signal and decision.timestamp <= latest_logged_signal:
            return BotIterationResult(action="idle", detail="Signal already processed for this bar.")

        self.repository.log_trade_event(
            TradeLogEvent(
                source=self.config.execution_mode.value,
                bot_run_id=self.bot_run_id,
                strategy_id=self.config.strategy_id,
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                event_type="signal",
                side=decision.action,
                size=decision.reference_price and (self.config.fixed_notional / decision.reference_price),
                status="signal",
                event_timestamp=decision.timestamp,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                notes=decision.rationale,
                raw_payload_json=decision.model_dump(),
            )
        )

        if has_open_position or has_open_orders:
            return BotIterationResult(action="idle", detail="Signal skipped because an existing position or order is active.")

        size = self.config.fixed_notional / decision.reference_price
        intent = OrderIntent(
            mode=self.config.execution_mode,
            symbol=self.config.symbol,
            action=decision.action,
            size=size,
            reference_price=decision.reference_price,
            entry_mode=EntryMode.MARKET,
            leverage=self.config.leverage,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            confidence=decision.confidence,
            thesis_summary=decision.thesis_summary,
            time_horizon=f"{self.config.timeframe} intraday",
            invalidation=decision.invalidation,
            decision_timestamp=decision.timestamp,
            rationale=decision.rationale,
        )

        if self.config.execution_mode == ExecutionMode.PAPER:
            self._submit_paper_order(intent)
            return BotIterationResult(action="paper_entry", detail=f"Opened paper {intent.action.value.lower()} position.")

        preview = self.executor.execute(intent)
        self.repository.log_trade_event(
            TradeLogEvent(
                source="live",
                bot_run_id=self.bot_run_id,
                strategy_id=self.config.strategy_id,
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                event_type="order_submitted",
                side=intent.action,
                size=intent.size,
                status=preview.status.value,
                event_timestamp=utc_now(),
                entry_timestamp=decision.timestamp,
                entry_price=intent.reference_price,
                stop_loss=intent.stop_loss,
                take_profit=intent.take_profit,
                order_id=preview.order_id,
                notes=preview.message,
                raw_payload_json=preview.model_dump(),
            )
        )
        return BotIterationResult(action="live_order", detail=preview.message)

    def run_forever(self) -> None:
        while True:
            self.run_once()
            time.sleep(self.config.poll_interval_seconds)

    def _load_completed_candles(self) -> pd.DataFrame:
        now = datetime.now(UTC)
        frame = self.executor.get_historical_ohlcv(
            self.config.symbol,
            start_time=_lookback_start(now, self.config.timeframe),
            end_time=now.isoformat(),
            timeframe=self.config.timeframe,
        )
        if frame.empty:
            return frame
        if "Date" in frame.columns:
            frame = frame.set_index("Date")
        frame.index = pd.DatetimeIndex(frame.index, tz="UTC")
        cutoff = _floor_time(pd.Timestamp(now), self.config.timeframe)
        return frame.loc[frame.index < cutoff].sort_index()

    def _submit_paper_order(self, intent: OrderIntent) -> None:
        payload = {
            "reference_price": intent.reference_price,
            "time_stop_bars": self.config.time_stop_bars,
        }
        self.repository.log_trade_event(
            TradeLogEvent(
                source="paper",
                bot_run_id=self.bot_run_id,
                strategy_id=self.config.strategy_id,
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                event_type="paper_entry",
                side=intent.action,
                size=intent.size,
                status=OrderStatus.FILLED.value,
                event_timestamp=utc_now(),
                entry_timestamp=utc_now(),
                entry_price=intent.reference_price,
                stop_loss=intent.stop_loss,
                take_profit=intent.take_profit,
                raw_payload_json=payload,
            )
        )

    def _sync_paper_position(self, completed: pd.DataFrame) -> Optional[BotIterationResult]:
        open_trade = self.repository.get_open_trade(source="paper", symbol=self.config.symbol)
        if open_trade is None:
            return None

        entry_timestamp = pd.Timestamp(open_trade["entry_timestamp"], tz="UTC")
        bars_since_entry = completed.loc[completed.index >= entry_timestamp]
        if bars_since_entry.empty:
            return None

        side = TradeAction(open_trade["side"])
        stop = float(open_trade["stop_loss"])
        take_profit = float(open_trade["take_profit"])
        exit_price = None
        reason = None
        latest_bar = bars_since_entry.iloc[-1]
        if side == TradeAction.LONG:
            if float(latest_bar["Low"]) <= stop:
                exit_price = stop
                reason = "paper stop hit"
            elif float(latest_bar["High"]) >= take_profit:
                exit_price = take_profit
                reason = "paper take profit hit"
        else:
            if float(latest_bar["High"]) >= stop:
                exit_price = stop
                reason = "paper stop hit"
            elif float(latest_bar["Low"]) <= take_profit:
                exit_price = take_profit
                reason = "paper take profit hit"

        if exit_price is None and len(bars_since_entry) >= self.config.time_stop_bars:
            exit_price = float(latest_bar["Close"])
            reason = "paper time stop"

        if exit_price is None:
            return None

        entry_price = float(open_trade["entry_price"])
        size = float(open_trade["size"])
        pnl = (exit_price - entry_price) * size if side == TradeAction.LONG else (entry_price - exit_price) * size
        self.repository.close_trade_event(
            int(open_trade["id"]),
            exit_timestamp=str(completed.index[-1].isoformat()),
            exit_price=exit_price,
            pnl=pnl,
            notes=reason,
        )
        self.repository.log_trade_event(
            TradeLogEvent(
                source="paper",
                bot_run_id=self.bot_run_id,
                strategy_id=self.config.strategy_id,
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                event_type="paper_exit",
                side=side,
                size=size,
                status="closed",
                event_timestamp=str(completed.index[-1].isoformat()),
                entry_timestamp=open_trade["entry_timestamp"],
                exit_timestamp=str(completed.index[-1].isoformat()),
                entry_price=entry_price,
                exit_price=exit_price,
                stop_loss=stop,
                take_profit=take_profit,
                pnl=pnl,
                notes=reason,
            )
        )
        return BotIterationResult(action="paper_exit", detail=reason)

    def _sync_live_position(
        self,
        snapshot: ExchangeStateSnapshot,
        completed: pd.DataFrame,
    ) -> Optional[BotIterationResult]:
        open_trade = self.repository.get_open_trade(source="live", symbol=self.config.symbol)
        if snapshot.positions and open_trade is None:
            position = snapshot.positions[0]
            self.repository.log_trade_event(
                TradeLogEvent(
                    source="live",
                    bot_run_id=self.bot_run_id,
                    strategy_id=self.config.strategy_id,
                    symbol=self.config.symbol,
                    timeframe=self.config.timeframe,
                    event_type="position_opened",
                    side=position.side,
                    size=position.size,
                    status="open",
                    event_timestamp=utc_now(),
                    entry_timestamp=utc_now(),
                    entry_price=position.entry_price,
                )
            )
            return BotIterationResult(action="sync", detail="Logged newly opened live position.")

        if not snapshot.positions and open_trade is not None:
            exit_price = snapshot.mark_prices.get(self.config.symbol) or snapshot.mark_prices.get(self.config.symbol.replace("-USD", ""))
            if exit_price is None:
                exit_price = float(open_trade["entry_price"])
            side = TradeAction(open_trade["side"])
            entry_price = float(open_trade["entry_price"])
            size = float(open_trade["size"])
            pnl = (exit_price - entry_price) * size if side == TradeAction.LONG else (entry_price - exit_price) * size
            self.repository.close_trade_event(
                int(open_trade["id"]),
                exit_timestamp=utc_now(),
                exit_price=float(exit_price),
                pnl=pnl,
                notes="live sync close",
            )
            return BotIterationResult(action="sync", detail="Closed stale live position record from exchange state.")

        if snapshot.positions and open_trade is not None:
            entry_timestamp = pd.Timestamp(open_trade["entry_timestamp"], tz="UTC")
            bars_since_entry = completed.loc[completed.index >= entry_timestamp]
            if len(bars_since_entry) >= self.config.time_stop_bars:
                preview = self.executor.execute(
                    OrderIntent(
                        mode=ExecutionMode.LIVE,
                        symbol=self.config.symbol,
                        action=TradeAction.FLAT,
                        size=float(open_trade["size"]),
                        reference_price=float(snapshot.positions[0].entry_price),
                        entry_mode=EntryMode.MARKET,
                        leverage=self.config.leverage,
                        confidence=1.0,
                        thesis_summary="Time stop reached.",
                        time_horizon=f"{self.config.timeframe} intraday",
                        invalidation="Time stop",
                        decision_timestamp=utc_now(),
                        rationale="Force close because the configured time stop elapsed.",
                        reduce_only=True,
                    )
                )
                self.repository.log_trade_event(
                    TradeLogEvent(
                        source="live",
                        bot_run_id=self.bot_run_id,
                        strategy_id=self.config.strategy_id,
                        symbol=self.config.symbol,
                        timeframe=self.config.timeframe,
                        event_type="time_stop_close",
                        side=TradeAction.FLAT,
                        size=float(open_trade["size"]),
                        status=preview.status.value,
                        event_timestamp=utc_now(),
                        order_id=preview.order_id,
                        notes=preview.message,
                        raw_payload_json=preview.model_dump(),
                    )
                )
                return BotIterationResult(action="live_close", detail=preview.message)
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Hyperliquid intraday trading bot.")
    parser.add_argument("--db-path", default="trading_data/trading.sqlite3")
    parser.add_argument("--strategy-id", default="trend_pullback")
    parser.add_argument("--symbol", default="BTC-USD")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--mode", choices=[mode.value for mode in ExecutionMode], default=ExecutionMode.PAPER.value)
    parser.add_argument("--fixed-notional", type=float, default=100.0)
    parser.add_argument("--leverage", type=int, default=1)
    parser.add_argument("--time-stop-bars", type=int, default=12)
    parser.add_argument("--poll-interval-seconds", type=int, default=30)
    parser.add_argument("--testnet", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    repository = TradingRepository(args.db_path)
    config = BotConfig(
        strategy_id=args.strategy_id,
        symbol=args.symbol,
        timeframe=args.timeframe,
        execution_mode=ExecutionMode(args.mode),
        fixed_notional=args.fixed_notional,
        leverage=args.leverage,
        time_stop_bars=args.time_stop_bars,
        poll_interval_seconds=args.poll_interval_seconds,
    )

    executor = HyperliquidExecutor(testnet=args.testnet)

    bot_run = repository.start_bot_run(
        mode=config.execution_mode.value,
        strategy_id=config.strategy_id,
        symbol=config.symbol,
        timeframe=config.timeframe,
        config=config.model_dump(mode="json"),
    )
    bot = TradingBot(
        config=config,
        repository=repository,
        executor=executor,
        bot_run_id=bot_run.id,
    )

    try:
        if args.once:
            result = bot.run_once()
            print(f"{result.action}: {result.detail}")
        else:
            bot.run_forever()
    except HyperliquidExecutionError as exc:
        repository.stop_bot_run(bot_run.id, status="failed", error=str(exc))
        raise
    except KeyboardInterrupt:
        repository.stop_bot_run(bot_run.id, status="stopped")
    else:
        repository.stop_bot_run(bot_run.id, status="completed")


if __name__ == "__main__":
    main()
