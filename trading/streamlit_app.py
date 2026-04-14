from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pandas as pd
import requests

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading.db import TradingRepository
from trading.market_data import HyperliquidMarketDataClient
from trading.models import BacktestRequest
from trading.services import BacktestService
from trading.strategy_registry import build_default_registry


def _load_streamlit():
    try:
        st = importlib.import_module("streamlit")
        components = importlib.import_module("streamlit.components.v1")
        return st, components
    except ModuleNotFoundError as exc:
        raise RuntimeError("Streamlit is not installed. Install the trading dependencies first.") from exc


def main() -> None:
    st, components = _load_streamlit()
    registry = build_default_registry()
    db_path = Path(os.getenv("TRADING_DB_PATH", "trading_data/trading.sqlite3"))
    artifacts_dir = Path(os.getenv("TRADING_ARTIFACTS_DIR", "trading_data/artifacts"))
    repository = TradingRepository(db_path)

    st.set_page_config(page_title="Backtesting.py Trading UI", layout="wide")
    st.title("Backtesting.py Trading UI")

    page = st.sidebar.radio("View", ["Strategy Catalog", "Backtest Runner", "History"])

    if page == "Strategy Catalog":
        for metadata in registry.list_metadata():
            with st.container(border=True):
                st.subheader(metadata.name)
                st.write(metadata.description)
                st.caption(f"Markets: {', '.join(metadata.supported_markets)} | Timeframes: {', '.join(metadata.supported_timeframes)}")
                schema = pd.DataFrame([item.model_dump() for item in metadata.param_schema])
                st.dataframe(schema, use_container_width=True, hide_index=True)
    elif page == "Backtest Runner":
        strategy_id = st.selectbox(
            "Strategy",
            options=[metadata.id for metadata in registry.list_metadata()],
            format_func=lambda value: registry.get(value).metadata.name,
        )
        definition = registry.get(strategy_id)
        with st.form("backtest-form"):
            symbol = st.selectbox("Market", definition.metadata.supported_markets)
            timeframe = st.selectbox("Timeframe", definition.metadata.supported_timeframes)
            start_time = st.text_input("Start Time (UTC ISO8601)", "2025-01-01T00:00:00+00:00")
            end_time = st.text_input("End Time (UTC ISO8601)", "2025-02-01T00:00:00+00:00")
            cash = st.number_input("Starting Cash", min_value=100.0, value=10000.0, step=100.0)
            commission = st.number_input("Commission", min_value=0.0, value=0.0005, step=0.0001, format="%.4f")
            spread = st.number_input("Spread", min_value=0.0, value=0.0, step=0.0001, format="%.4f")
            params = {}
            for spec in definition.metadata.param_schema:
                if spec.type == "integer":
                    params[spec.name] = st.number_input(
                        spec.label,
                        min_value=int(spec.minimum or 0),
                        max_value=int(spec.maximum or 500),
                        value=int(spec.default),
                        step=int(spec.step or 1),
                        help=spec.description,
                    )
                else:
                    params[spec.name] = st.number_input(
                        spec.label,
                        min_value=float(spec.minimum or 0.0),
                        max_value=float(spec.maximum or 100.0),
                        value=float(spec.default),
                        step=float(spec.step or 0.1),
                        help=spec.description,
                    )
            submitted = st.form_submit_button("Run Backtest")

        if submitted:
            try:
                market_data = HyperliquidMarketDataClient(testnet=False)
                service = BacktestService(
                    repository=repository,
                    registry=registry,
                    market_data=market_data,
                    artifacts_dir=artifacts_dir,
                )
                result = service.run_backtest(
                    BacktestRequest(
                        strategy_id=strategy_id,
                        symbol=symbol,
                        timeframe=timeframe,
                        start_time=start_time,
                        end_time=end_time,
                        cash=cash,
                        commission=commission,
                        spread=spread,
                        params=params,
                    )
                )
            except requests.RequestException as exc:
                st.error(f"Failed to reach Hyperliquid market data: {exc}")
            except Exception as exc:
                st.error(str(exc))
            else:
                st.success(f"Backtest run #{result.run_id} completed.")
                metrics_df = pd.DataFrame(list(result.metrics.items()), columns=["Metric", "Value"])
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                if result.trades:
                    st.subheader("Trades")
                    st.dataframe(pd.DataFrame(result.trades), use_container_width=True)
                if result.plot_path:
                    plot_html = Path(result.plot_path).read_text(encoding="utf-8")
                    components.html(plot_html, height=900, scrolling=True)
    else:
        st.subheader("Backtest Runs")
        backtests = repository.list_backtest_runs()
        if backtests:
            st.dataframe(pd.DataFrame(backtests), use_container_width=True)
        else:
            st.info("No backtest runs recorded yet.")

        st.subheader("Bot Runs")
        bot_runs = repository.list_bot_runs()
        if bot_runs:
            st.dataframe(pd.DataFrame(bot_runs), use_container_width=True)
        else:
            st.info("No bot runs recorded yet.")

        st.subheader("Trade Log")
        trade_events = repository.list_trade_events()
        if trade_events:
            st.dataframe(pd.DataFrame(trade_events), use_container_width=True)
        else:
            st.info("No trade events recorded yet.")


if __name__ == "__main__":
    main()
