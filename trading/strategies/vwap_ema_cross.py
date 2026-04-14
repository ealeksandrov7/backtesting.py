from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from trading.models import StrategyDecision, TradeAction
from trading.strategies.trend_pullback import _atr, _dmi_adx, _ema, _resample_ohlcv, _rsi, _session_vwap


class VwapEmaCrossParams(BaseModel):
    ema_fast: int = Field(default=9, ge=2, le=100)
    ema_slow: int = Field(default=21, ge=3, le=200)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiplier: float = Field(default=2.2, gt=0.1, le=10.0)
    reward_to_risk: float = Field(default=3.5, gt=0.1, le=10.0)
    time_stop_bars: int = Field(default=36, ge=1, le=200)
    higher_tf_validation: bool = True
    allow_long: bool = True
    allow_short: bool = True
    use_4h_confirmation: bool = True
    higher_tf_4h_fast: int = Field(default=50, ge=2, le=200)
    higher_tf_4h_slow: int = Field(default=200, ge=3, le=400)
    higher_tf_adx_period: int = Field(default=14, ge=2, le=100)
    higher_tf_adx_threshold: float = Field(default=35.0, ge=0.0, le=100.0)
    require_htf_adx: bool = True
    rsi_period: int = Field(default=14, ge=2, le=100)
    rsi_threshold: float = Field(default=50.0, ge=0.0, le=100.0)
    swing_lookback: int = Field(default=3, ge=1, le=50)
    break_even_enabled: bool = False
    break_even_reward_multiple: float = Field(default=1.0, ge=0.1, le=10.0)


def compute_vwap_ema_cross_frame(
    data: pd.DataFrame,
    params: VwapEmaCrossParams | dict | None = None,
) -> pd.DataFrame:
    params = params if isinstance(params, VwapEmaCrossParams) else VwapEmaCrossParams(**(params or {}))

    frame = data.copy()
    if "Volume" not in frame.columns:
        frame["Volume"] = 1.0

    frame["ema_fast"] = _ema(frame["Close"], params.ema_fast)
    frame["ema_slow"] = _ema(frame["Close"], params.ema_slow)
    frame["atr"] = _atr(frame, params.atr_period)
    frame["session_vwap"] = _session_vwap(frame)
    frame["rsi"] = _rsi(frame["Close"], params.rsi_period)

    long_htf_regime = pd.Series(True, index=frame.index, dtype=bool)
    short_htf_regime = pd.Series(True, index=frame.index, dtype=bool)
    if params.higher_tf_validation and isinstance(frame.index, pd.DatetimeIndex) and params.use_4h_confirmation:
        frame_4h = _resample_ohlcv(frame[["Open", "High", "Low", "Close", "Volume"]], "4h")
        plus_di_4h, minus_di_4h, adx_4h = _dmi_adx(frame_4h, params.higher_tf_adx_period)
        frame["htf_4h_fast"] = _ema(frame_4h["Close"], params.higher_tf_4h_fast).reindex(
            frame.index, method="ffill"
        )
        frame["htf_4h_slow"] = _ema(frame_4h["Close"], params.higher_tf_4h_slow).reindex(
            frame.index, method="ffill"
        )
        frame["htf_4h_plus_di"] = plus_di_4h.reindex(frame.index, method="ffill")
        frame["htf_4h_minus_di"] = minus_di_4h.reindex(frame.index, method="ffill")
        frame["htf_4h_adx"] = adx_4h.reindex(frame.index, method="ffill")

        long_htf_regime = (
            (frame["htf_4h_fast"] > frame["htf_4h_slow"])
            & (frame["htf_4h_plus_di"] > frame["htf_4h_minus_di"])
        ).fillna(False)
        short_htf_regime = (
            (frame["htf_4h_fast"] < frame["htf_4h_slow"])
            & (frame["htf_4h_minus_di"] > frame["htf_4h_plus_di"])
        ).fillna(False)
        if params.require_htf_adx:
            long_htf_regime &= (frame["htf_4h_adx"] >= params.higher_tf_adx_threshold).fillna(False)
            short_htf_regime &= (frame["htf_4h_adx"] >= params.higher_tf_adx_threshold).fillna(False)

    cross_up = (frame["ema_fast"] > frame["ema_slow"]) & (frame["ema_fast"].shift(1) <= frame["ema_slow"].shift(1))
    cross_down = (frame["ema_fast"] < frame["ema_slow"]) & (frame["ema_fast"].shift(1) >= frame["ema_slow"].shift(1))

    frame["long_signal"] = (
        long_htf_regime
        & cross_up
        & (frame["Close"] > frame["session_vwap"])
        & (frame["rsi"] >= params.rsi_threshold)
    )
    frame["short_signal"] = (
        short_htf_regime
        & cross_down
        & (frame["Close"] < frame["session_vwap"])
        & (frame["rsi"] <= (100 - params.rsi_threshold))
    )

    frame["swing_low"] = frame["Low"].shift(1).rolling(params.swing_lookback, min_periods=1).min()
    frame["swing_high"] = frame["High"].shift(1).rolling(params.swing_lookback, min_periods=1).max()
    atr_long_stop = frame["Close"] - (frame["atr"] * params.atr_stop_multiplier)
    atr_short_stop = frame["Close"] + (frame["atr"] * params.atr_stop_multiplier)
    frame["long_stop"] = np.minimum(frame["swing_low"], atr_long_stop)
    frame["short_stop"] = np.maximum(frame["swing_high"], atr_short_stop)
    frame["long_risk"] = frame["Close"] - frame["long_stop"]
    frame["short_risk"] = frame["short_stop"] - frame["Close"]
    frame["long_take_profit"] = frame["Close"] + (frame["long_risk"] * params.reward_to_risk)
    frame["short_take_profit"] = frame["Close"] - (frame["short_risk"] * params.reward_to_risk)

    frame.loc[frame["long_risk"] <= 0, ["long_signal", "long_stop", "long_take_profit"]] = [False, np.nan, np.nan]
    frame.loc[frame["short_risk"] <= 0, ["short_signal", "short_stop", "short_take_profit"]] = [False, np.nan, np.nan]

    frame["signal"] = 0
    frame.loc[frame["long_signal"], "signal"] = 1
    frame.loc[frame["short_signal"], "signal"] = -1

    if not params.allow_long:
        frame.loc[frame["signal"] == 1, "signal"] = 0
        frame.loc[:, ["long_signal", "long_stop", "long_take_profit"]] = [False, np.nan, np.nan]
    if not params.allow_short:
        frame.loc[frame["signal"] == -1, "signal"] = 0
        frame.loc[:, ["short_signal", "short_stop", "short_take_profit"]] = [False, np.nan, np.nan]

    return frame


def latest_vwap_ema_cross_decision(
    data: pd.DataFrame,
    symbol: str,
    timeframe: str,
    params: VwapEmaCrossParams | dict | None = None,
) -> Optional[StrategyDecision]:
    if data.empty:
        return None

    params = params if isinstance(params, VwapEmaCrossParams) else VwapEmaCrossParams(**(params or {}))
    frame = compute_vwap_ema_cross_frame(data, params)
    row = frame.iloc[-1]
    timestamp = frame.index[-1] if frame.index.name or isinstance(frame.index, pd.DatetimeIndex) else data.iloc[-1].get("Date")
    if pd.isna(timestamp):
        timestamp = pd.Timestamp.utcnow()

    if row["signal"] == 1:
        return StrategyDecision(
            strategy_id="vwap_ema_cross",
            symbol=symbol,
            timeframe=timeframe,
            timestamp=pd.Timestamp(timestamp).isoformat(),
            action=TradeAction.LONG,
            reference_price=float(row["Close"]),
            stop_loss=float(row["long_stop"]),
            take_profit=float(row["long_take_profit"]),
            confidence=0.56,
            thesis_summary="BTC 15m bullish EMA cross confirmed above session VWAP in a strong 4h uptrend.",
            invalidation="Price loses the recent swing low or ATR-based stop.",
            rationale="The 9 EMA crossed above the 21 EMA while price stayed above session VWAP and the 4h trend filter remained bullish.",
            time_stop_bars=params.time_stop_bars,
        )
    if row["signal"] == -1:
        return StrategyDecision(
            strategy_id="vwap_ema_cross",
            symbol=symbol,
            timeframe=timeframe,
            timestamp=pd.Timestamp(timestamp).isoformat(),
            action=TradeAction.SHORT,
            reference_price=float(row["Close"]),
            stop_loss=float(row["short_stop"]),
            take_profit=float(row["short_take_profit"]),
            confidence=0.56,
            thesis_summary="BTC 15m bearish EMA cross confirmed below session VWAP in a strong 4h downtrend.",
            invalidation="Price reclaims the recent swing high or ATR-based stop.",
            rationale="The 9 EMA crossed below the 21 EMA while price stayed below session VWAP and the 4h trend filter remained bearish.",
            time_stop_bars=params.time_stop_bars,
        )
    return None


def build_vwap_ema_cross_strategy_class():
    from backtesting import Strategy

    def _writable_array(series: pd.Series) -> np.ndarray:
        return np.array(series.to_numpy(copy=True), copy=True)

    class VwapEmaCrossStrategy(Strategy):
        ema_fast = 9
        ema_slow = 21
        atr_period = 14
        atr_stop_multiplier = 2.2
        reward_to_risk = 3.5
        time_stop_bars = 36
        higher_tf_validation = True
        allow_long = True
        allow_short = True
        use_4h_confirmation = True
        higher_tf_4h_fast = 50
        higher_tf_4h_slow = 200
        higher_tf_adx_period = 14
        higher_tf_adx_threshold = 35.0
        require_htf_adx = True
        rsi_period = 14
        rsi_threshold = 50.0
        swing_lookback = 3
        break_even_enabled = False
        break_even_reward_multiple = 1.0

        def init(self):
            data = pd.DataFrame(
                {
                    "Open": pd.Series(self.data.Open, index=self.data.index),
                    "High": pd.Series(self.data.High, index=self.data.index),
                    "Low": pd.Series(self.data.Low, index=self.data.index),
                    "Close": pd.Series(self.data.Close, index=self.data.index),
                    "Volume": pd.Series(self.data.Volume, index=self.data.index),
                }
            )
            params = VwapEmaCrossParams(
                ema_fast=self.ema_fast,
                ema_slow=self.ema_slow,
                atr_period=self.atr_period,
                atr_stop_multiplier=self.atr_stop_multiplier,
                reward_to_risk=self.reward_to_risk,
                time_stop_bars=self.time_stop_bars,
                higher_tf_validation=self.higher_tf_validation,
                allow_long=self.allow_long,
                allow_short=self.allow_short,
                use_4h_confirmation=self.use_4h_confirmation,
                higher_tf_4h_fast=self.higher_tf_4h_fast,
                higher_tf_4h_slow=self.higher_tf_4h_slow,
                higher_tf_adx_period=self.higher_tf_adx_period,
                higher_tf_adx_threshold=self.higher_tf_adx_threshold,
                require_htf_adx=self.require_htf_adx,
                rsi_period=self.rsi_period,
                rsi_threshold=self.rsi_threshold,
                swing_lookback=self.swing_lookback,
                break_even_enabled=self.break_even_enabled,
                break_even_reward_multiple=self.break_even_reward_multiple,
            )
            signal_frame = compute_vwap_ema_cross_frame(data, params)
            self.signal = self.I(lambda: _writable_array(signal_frame["signal"]), plot=False)
            self.long_stop = self.I(lambda: _writable_array(signal_frame["long_stop"]), plot=False)
            self.short_stop = self.I(lambda: _writable_array(signal_frame["short_stop"]), plot=False)
            self.long_take_profit = self.I(lambda: _writable_array(signal_frame["long_take_profit"]), plot=False)
            self.short_take_profit = self.I(lambda: _writable_array(signal_frame["short_take_profit"]), plot=False)

        def next(self):
            if self.position and self.trades:
                active_trade = self.trades[-1]
                if self.break_even_enabled:
                    entry_bar = active_trade.entry_bar
                    if active_trade.is_long:
                        initial_stop = float(self.long_stop[entry_bar])
                        initial_risk = float(active_trade.entry_price) - initial_stop
                        if (
                            np.isfinite(initial_risk)
                            and initial_risk > 0
                            and float(self.data.High[-1]) >= float(active_trade.entry_price) + (initial_risk * self.break_even_reward_multiple)
                        ):
                            active_trade.sl = max(float(active_trade.sl or -np.inf), float(active_trade.entry_price))
                    else:
                        initial_stop = float(self.short_stop[entry_bar])
                        initial_risk = initial_stop - float(active_trade.entry_price)
                        if (
                            np.isfinite(initial_risk)
                            and initial_risk > 0
                            and float(self.data.Low[-1]) <= float(active_trade.entry_price) - (initial_risk * self.break_even_reward_multiple)
                        ):
                            active_trade.sl = min(float(active_trade.sl or np.inf), float(active_trade.entry_price))
                if len(self.data) - 1 - active_trade.entry_bar >= self.time_stop_bars:
                    self.position.close()
                    return

            if self.position:
                return

            if self.signal[-1] == 1:
                self.buy(sl=float(self.long_stop[-1]), tp=float(self.long_take_profit[-1]))
            elif self.signal[-1] == -1:
                self.sell(sl=float(self.short_stop[-1]), tp=float(self.short_take_profit[-1]))

    VwapEmaCrossStrategy.__name__ = "VwapEmaCrossStrategy"
    return VwapEmaCrossStrategy
