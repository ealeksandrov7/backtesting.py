from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from trading.models import BotRunRecord, TradeLogEvent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TradingRepository:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    cash REAL NOT NULL,
                    commission REAL NOT NULL,
                    spread REAL NOT NULL,
                    params_json TEXT NOT NULL,
                    metrics_json TEXT,
                    plot_path TEXT,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS bot_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mode TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    stopped_at TEXT,
                    last_heartbeat TEXT,
                    status TEXT NOT NULL,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS trade_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    run_id INTEGER,
                    bot_run_id INTEGER,
                    strategy_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT,
                    event_type TEXT NOT NULL,
                    side TEXT,
                    size REAL,
                    status TEXT,
                    event_timestamp TEXT NOT NULL,
                    entry_timestamp TEXT,
                    exit_timestamp TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    pnl REAL,
                    fees REAL,
                    order_id TEXT,
                    notes TEXT,
                    raw_payload_json TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )

    def create_backtest_run(self, *, strategy_id: str, symbol: str, timeframe: str, start_time: str, end_time: str, cash: float, commission: float, spread: float, params: dict) -> int:
        now = utc_now()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO backtest_runs (
                    strategy_id, symbol, timeframe, start_time, end_time,
                    cash, commission, spread, params_json, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    strategy_id,
                    symbol,
                    timeframe,
                    start_time,
                    end_time,
                    cash,
                    commission,
                    spread,
                    json.dumps(params),
                    "running",
                    now,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def complete_backtest_run(self, run_id: int, *, metrics: dict[str, Any], plot_path: Optional[str]) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE backtest_runs
                SET metrics_json = ?, plot_path = ?, status = ?, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(metrics), plot_path, "completed", now, run_id),
            )

    def fail_backtest_run(self, run_id: int, *, error: str) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE backtest_runs
                SET status = ?, error = ?, updated_at = ?
                WHERE id = ?
                """,
                ("failed", error, now, run_id),
            )

    def list_backtest_runs(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM backtest_runs ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def start_bot_run(self, *, mode: str, strategy_id: str, symbol: str, timeframe: str, config: dict[str, Any]) -> BotRunRecord:
        now = utc_now()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO bot_runs (
                    mode, strategy_id, symbol, timeframe, config_json,
                    started_at, last_heartbeat, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (mode, strategy_id, symbol, timeframe, json.dumps(config), now, now, "running"),
            )
            run_id = int(cursor.lastrowid)
        return BotRunRecord(
            id=run_id,
            mode=mode,
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            config_json=config,
            started_at=now,
            last_heartbeat=now,
            status="running",
        )

    def heartbeat_bot_run(self, bot_run_id: int) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                "UPDATE bot_runs SET last_heartbeat = ? WHERE id = ?",
                (now, bot_run_id),
            )

    def stop_bot_run(self, bot_run_id: int, *, status: str, error: Optional[str] = None) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE bot_runs
                SET stopped_at = ?, last_heartbeat = ?, status = ?, error = ?
                WHERE id = ?
                """,
                (now, now, status, error, bot_run_id),
            )

    def list_bot_runs(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM bot_runs ORDER BY started_at DESC"
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def log_trade_event(self, event: TradeLogEvent) -> int:
        payload = event.model_dump()
        payload["raw_payload_json"] = json.dumps(payload["raw_payload_json"]) if payload["raw_payload_json"] is not None else None
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO trade_log (
                    source, run_id, bot_run_id, strategy_id, symbol, timeframe,
                    event_type, side, size, status, event_timestamp, entry_timestamp,
                    exit_timestamp, entry_price, exit_price, stop_loss, take_profit,
                    pnl, fees, order_id, notes, raw_payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["source"],
                    payload["run_id"],
                    payload["bot_run_id"],
                    payload["strategy_id"],
                    payload["symbol"],
                    payload["timeframe"],
                    payload["event_type"],
                    payload["side"],
                    payload["size"],
                    payload["status"],
                    payload["event_timestamp"],
                    payload["entry_timestamp"],
                    payload["exit_timestamp"],
                    payload["entry_price"],
                    payload["exit_price"],
                    payload["stop_loss"],
                    payload["take_profit"],
                    payload["pnl"],
                    payload["fees"],
                    payload["order_id"],
                    payload["notes"],
                    payload["raw_payload_json"],
                    payload["created_at"],
                ),
            )
            return int(cursor.lastrowid)

    def list_trade_events(self, *, limit: int = 200) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM trade_log ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def latest_signal_timestamp(self, *, bot_run_id: int) -> Optional[str]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT event_timestamp
                FROM trade_log
                WHERE bot_run_id = ? AND event_type IN ('signal', 'order_submitted', 'paper_entry')
                ORDER BY event_timestamp DESC
                LIMIT 1
                """,
                (bot_run_id,),
            ).fetchone()
        return None if row is None else str(row["event_timestamp"])

    def get_open_trade(self, *, source: str, symbol: str) -> Optional[dict[str, Any]]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM trade_log
                WHERE source = ? AND symbol = ? AND event_type IN ('paper_entry', 'position_opened')
                AND exit_timestamp IS NULL
                ORDER BY event_timestamp DESC
                LIMIT 1
                """,
                (source, symbol),
            ).fetchone()
        return None if row is None else self._row_to_dict(row)

    def close_trade_event(self, event_id: int, *, exit_timestamp: str, exit_price: float, pnl: float, notes: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE trade_log
                SET exit_timestamp = ?, exit_price = ?, pnl = ?, notes = COALESCE(notes, '') || ?, status = ?
                WHERE id = ?
                """,
                (exit_timestamp, exit_price, pnl, f"\n{notes}", "closed", event_id),
            )

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        for key in ("config_json", "params_json", "metrics_json", "raw_payload_json"):
            value = result.get(key)
            if value:
                result[key] = json.loads(value)
        return result
