from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from trading.models import StrategyDecision, TradeAction


class TrendPullbackParams(BaseModel):
    ema_fast: int = Field(default=12, ge=2, le=200)
    ema_trend: int = Field(default=30, ge=3, le=300)
    ema_bias: int = Field(default=120, ge=5, le=500)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiplier: float = Field(default=1.2, gt=0.1, le=10.0)
    reward_to_risk: float = Field(default=3.0, gt=0.1, le=10.0)
    time_stop_bars: int = Field(default=20, ge=1, le=200)
    higher_tf_validation: bool = True
    allow_long: bool = True
    allow_short: bool = True
    use_4h_confirmation: bool = True
    higher_tf_4h_fast: int = Field(default=20, ge=2, le=200)
    higher_tf_4h_slow: int = Field(default=50, ge=3, le=300)
    higher_tf_adx_period: int = Field(default=14, ge=2, le=100)
    higher_tf_adx_threshold: float = Field(default=28.0, ge=0.0, le=100.0)
    higher_tf_slope_bars: int = Field(default=3, ge=1, le=50)
    rsi_period: int = Field(default=14, ge=2, le=100)
    rsi_trend_threshold: float = Field(default=51.0, ge=0.0, le=100.0)
    rsi_long_pullback_floor: float = Field(default=38.0, ge=0.0, le=100.0)
    rsi_long_pullback_ceiling: float = Field(default=50.0, ge=0.0, le=100.0)
    rsi_short_pullback_floor: float = Field(default=50.0, ge=0.0, le=100.0)
    rsi_short_pullback_ceiling: float = Field(default=62.0, ge=0.0, le=100.0)
    rsi_turn_threshold: float = Field(default=2.0, ge=0.0, le=50.0)
    pullback_atr_buffer: float = Field(default=0.25, ge=0.0, le=5.0)
    trigger_lookback: int = Field(default=1, ge=1, le=20)
    volume_window: int = Field(default=20, ge=2, le=200)
    volume_ratio_threshold: float = Field(default=1.0, gt=0.0, le=10.0)
    require_volume_confirmation: bool = False
    require_vwap_confirmation: bool = True
    trailing_stop_enabled: bool = False
    trailing_stop_lookback: int = Field(default=10, ge=2, le=200)
    trailing_stop_atr_multiplier: float = Field(default=2.5, gt=0.1, le=20.0)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(frame: pd.DataFrame, period: int) -> pd.Series:
    prev_close = frame["Close"].shift(1)
    ranges = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - prev_close).abs(),
            (frame["Low"] - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = ranges.max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def _session_vwap(frame: pd.DataFrame) -> pd.Series:
    typical_price = (frame["High"] + frame["Low"] + frame["Close"]) / 3.0
    if not isinstance(frame.index, pd.DatetimeIndex):
        cumulative_price_volume = (typical_price * frame["Volume"]).cumsum()
        cumulative_volume = frame["Volume"].cumsum()
        return cumulative_price_volume / cumulative_volume.replace(0, np.nan)

    timestamps = frame.index.tz_convert("UTC") if frame.index.tz is not None else frame.index
    sessions = pd.Series(timestamps.normalize(), index=frame.index)
    price_volume = typical_price * frame["Volume"]
    cumulative_price_volume = price_volume.groupby(sessions).cumsum()
    cumulative_volume = frame["Volume"].groupby(sessions).cumsum()
    return cumulative_price_volume / cumulative_volume.replace(0, np.nan)


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
    average_loss = losses.ewm(alpha=1 / period, adjust=False).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.where(average_loss != 0, 100.0)
    rsi = rsi.where(average_gain != 0, 0.0)
    return rsi


def _dmi_adx(frame: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = frame["High"].diff()
    down_move = -frame["Low"].diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=frame.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=frame.index,
    )
    atr = _atr(frame, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / di_sum) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return plus_di, minus_di, adx


def _resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        frame.resample(rule, label="right", closed="right")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )


def compute_trend_pullback_frame(
    data: pd.DataFrame,
    params: TrendPullbackParams | dict | None = None,
) -> pd.DataFrame:
    params = params if isinstance(params, TrendPullbackParams) else TrendPullbackParams(**(params or {}))

    frame = data.copy()
    if "Volume" not in frame.columns:
        frame["Volume"] = 1.0
    frame["ema_fast"] = _ema(frame["Close"], params.ema_fast)
    frame["ema_trend"] = _ema(frame["Close"], params.ema_trend)
    frame["ema_bias"] = _ema(frame["Close"], params.ema_bias)
    frame["atr"] = _atr(frame, params.atr_period)
    frame["volume_median"] = frame["Volume"].rolling(params.volume_window, min_periods=1).median()
    frame["session_vwap"] = _session_vwap(frame)
    frame["rsi"] = _rsi(frame["Close"], params.rsi_period)
    frame["trailing_high"] = frame["High"].rolling(params.trailing_stop_lookback, min_periods=1).max()
    frame["trailing_low"] = frame["Low"].rolling(params.trailing_stop_lookback, min_periods=1).min()
    frame["long_trailing_stop"] = (
        frame["trailing_high"] - (frame["atr"] * params.trailing_stop_atr_multiplier)
    )
    frame["short_trailing_stop"] = (
        frame["trailing_low"] + (frame["atr"] * params.trailing_stop_atr_multiplier)
    )

    long_htf_regime = pd.Series(True, index=frame.index, dtype=bool)
    short_htf_regime = pd.Series(True, index=frame.index, dtype=bool)
    if params.higher_tf_validation and isinstance(frame.index, pd.DatetimeIndex):
        if params.use_4h_confirmation:
            frame_4h = _resample_ohlcv(frame[["Open", "High", "Low", "Close", "Volume"]], "4h")
            plus_di_4h, minus_di_4h, adx_4h = _dmi_adx(frame_4h, params.higher_tf_adx_period)
            frame["htf_4h_close"] = frame_4h["Close"].reindex(frame.index, method="ffill")
            frame["htf_4h_fast"] = _ema(frame_4h["Close"], params.higher_tf_4h_fast).reindex(
                frame.index, method="ffill"
            )
            frame["htf_4h_slow"] = _ema(frame_4h["Close"], params.higher_tf_4h_slow).reindex(
                frame.index, method="ffill"
            )
            frame["htf_4h_slow_shift"] = _ema(frame_4h["Close"], params.higher_tf_4h_slow).shift(
                params.higher_tf_slope_bars
            ).reindex(frame.index, method="ffill")
            frame["htf_4h_plus_di"] = plus_di_4h.reindex(frame.index, method="ffill")
            frame["htf_4h_minus_di"] = minus_di_4h.reindex(frame.index, method="ffill")
            frame["htf_4h_adx"] = adx_4h.reindex(frame.index, method="ffill")
            long_htf_regime &= (
                (frame["htf_4h_close"] > frame["htf_4h_fast"])
                & (frame["htf_4h_fast"] > frame["htf_4h_slow"])
                & (frame["htf_4h_plus_di"] > frame["htf_4h_minus_di"])
                & (frame["htf_4h_adx"] >= params.higher_tf_adx_threshold)
                & (frame["htf_4h_slow"] > frame["htf_4h_slow_shift"])
            ).fillna(False)
            short_htf_regime &= (
                (frame["htf_4h_close"] < frame["htf_4h_fast"])
                & (frame["htf_4h_fast"] < frame["htf_4h_slow"])
                & (frame["htf_4h_minus_di"] > frame["htf_4h_plus_di"])
                & (frame["htf_4h_adx"] >= params.higher_tf_adx_threshold)
                & (frame["htf_4h_slow"] < frame["htf_4h_slow_shift"])
            ).fillna(False)

    long_regime = (
        (frame["ema_trend"] > frame["ema_bias"])
        & (frame["Close"] > frame["ema_trend"])
        & long_htf_regime
    )
    short_regime = (
        (frame["ema_trend"] < frame["ema_bias"])
        & (frame["Close"] < frame["ema_trend"])
        & short_htf_regime
    )

    long_pullback_floor = pd.concat([frame["ema_trend"], frame["session_vwap"]], axis=1).min(axis=1)
    long_pullback_ceiling = pd.concat([frame["ema_fast"], frame["session_vwap"]], axis=1).max(axis=1)
    short_pullback_floor = pd.concat([frame["ema_fast"], frame["session_vwap"]], axis=1).min(axis=1)
    short_pullback_ceiling = pd.concat([frame["ema_trend"], frame["session_vwap"]], axis=1).max(axis=1)
    pullback_buffer = frame["atr"] * params.pullback_atr_buffer

    prev_long_pullback = (
        (frame["Low"].shift(1) <= (long_pullback_ceiling + pullback_buffer).shift(1))
        & (
            frame["Close"].shift(1).between(
                (long_pullback_floor - pullback_buffer).shift(1),
                (long_pullback_ceiling + pullback_buffer).shift(1),
            )
        )
        & frame["rsi"].shift(1).between(
            params.rsi_long_pullback_floor,
            params.rsi_long_pullback_ceiling,
        )
    )
    prev_short_pullback = (
        (frame["High"].shift(1) >= (short_pullback_floor - pullback_buffer).shift(1))
        & (
            frame["Close"].shift(1).between(
                (short_pullback_floor - pullback_buffer).shift(1),
                (short_pullback_ceiling + pullback_buffer).shift(1),
            )
        )
        & frame["rsi"].shift(1).between(
            params.rsi_short_pullback_floor,
            params.rsi_short_pullback_ceiling,
        )
    )

    trigger_high = frame["High"].rolling(params.trigger_lookback, min_periods=1).max().shift(1)
    trigger_low = frame["Low"].rolling(params.trigger_lookback, min_periods=1).min().shift(1)

    bullish_confirmation = (
        (frame["Close"] > frame["Open"])
        & (frame["Close"] > frame["ema_fast"])
        & (frame["High"] > trigger_high)
        & ((frame["rsi"] - frame["rsi"].shift(1)) >= params.rsi_turn_threshold)
        & (frame["rsi"] >= params.rsi_trend_threshold)
    )
    bearish_confirmation = (
        (frame["Close"] < frame["Open"])
        & (frame["Close"] < frame["ema_fast"])
        & (frame["Low"] < trigger_low)
        & ((frame["rsi"].shift(1) - frame["rsi"]) >= params.rsi_turn_threshold)
        & (frame["rsi"] <= (100 - params.rsi_trend_threshold))
    )
    if params.require_vwap_confirmation:
        bullish_confirmation &= frame["Close"] > frame["session_vwap"]
        bearish_confirmation &= frame["Close"] < frame["session_vwap"]
    if params.require_volume_confirmation:
        bullish_confirmation &= frame["Volume"] >= frame["volume_median"] * params.volume_ratio_threshold
        bearish_confirmation &= frame["Volume"] >= frame["volume_median"] * params.volume_ratio_threshold

    frame["long_signal"] = (
        long_regime
        & long_regime.shift(1).fillna(False)
        & prev_long_pullback.fillna(False)
        & bullish_confirmation
    )
    frame["short_signal"] = (
        short_regime
        & short_regime.shift(1).fillna(False)
        & prev_short_pullback.fillna(False)
        & bearish_confirmation
    )

    frame["swing_low"] = frame["Low"].shift(1).rolling(3, min_periods=1).min()
    frame["swing_high"] = frame["High"].shift(1).rolling(3, min_periods=1).max()

    atr_long_stop = frame["Close"] - (frame["atr"] * params.atr_stop_multiplier)
    atr_short_stop = frame["Close"] + (frame["atr"] * params.atr_stop_multiplier)

    frame["long_stop"] = np.minimum(frame["swing_low"], atr_long_stop)
    frame["short_stop"] = np.maximum(frame["swing_high"], atr_short_stop)

    frame["long_risk"] = frame["Close"] - frame["long_stop"]
    frame["short_risk"] = frame["short_stop"] - frame["Close"]

    frame["long_take_profit"] = frame["Close"] + (frame["long_risk"] * params.reward_to_risk)
    frame["short_take_profit"] = frame["Close"] - (frame["short_risk"] * params.reward_to_risk)

    frame.loc[frame["long_risk"] <= 0, ["long_signal", "long_stop", "long_take_profit"]] = [
        False,
        np.nan,
        np.nan,
    ]
    frame.loc[frame["short_risk"] <= 0, ["short_signal", "short_stop", "short_take_profit"]] = [
        False,
        np.nan,
        np.nan,
    ]

    frame["signal"] = 0
    frame.loc[frame["long_signal"], "signal"] = 1
    frame.loc[frame["short_signal"], "signal"] = -1
    if not params.allow_long:
        frame.loc[frame["signal"] == 1, "signal"] = 0
        frame.loc[:, ["long_signal", "long_stop", "long_take_profit"]] = [
            False,
            np.nan,
            np.nan,
        ]
    if not params.allow_short:
        frame.loc[frame["signal"] == -1, "signal"] = 0
        frame.loc[:, ["short_signal", "short_stop", "short_take_profit"]] = [
            False,
            np.nan,
            np.nan,
        ]

    return frame


def latest_trend_pullback_decision(
    data: pd.DataFrame,
    symbol: str,
    timeframe: str,
    params: TrendPullbackParams | dict | None = None,
) -> Optional[StrategyDecision]:
    if data.empty:
        return None

    params = params if isinstance(params, TrendPullbackParams) else TrendPullbackParams(**(params or {}))
    frame = compute_trend_pullback_frame(data, params)
    row = frame.iloc[-1]
    timestamp = frame.index[-1] if frame.index.name or isinstance(frame.index, pd.DatetimeIndex) else data.iloc[-1].get("Date")
    if pd.isna(timestamp):
        timestamp = pd.Timestamp.utcnow()
    if row["signal"] == 1:
        return StrategyDecision(
            strategy_id="trend_pullback",
            symbol=symbol,
            timeframe=timeframe,
            timestamp=pd.Timestamp(timestamp).isoformat(),
            action=TradeAction.LONG,
            reference_price=float(row["Close"]),
            stop_loss=float(row["long_stop"]),
            take_profit=float(row["long_take_profit"]),
            confidence=0.6,
            thesis_summary="BTC long pullback aligned across 15m structure and higher-timeframe trend strength.",
            invalidation="Price loses the pullback swing low or ATR stop level.",
            rationale="The 15m pullback reclaimed EMA and session VWAP with RSI turning back up while the enabled 4h trend-strength filters remained bullish.",
            time_stop_bars=params.time_stop_bars,
        )
    if row["signal"] == -1:
        return StrategyDecision(
            strategy_id="trend_pullback",
            symbol=symbol,
            timeframe=timeframe,
            timestamp=pd.Timestamp(timestamp).isoformat(),
            action=TradeAction.SHORT,
            reference_price=float(row["Close"]),
            stop_loss=float(row["short_stop"]),
            take_profit=float(row["short_take_profit"]),
            confidence=0.6,
            thesis_summary="BTC short pullback aligned across 15m structure and higher-timeframe trend strength.",
            invalidation="Price reclaims the pullback swing high or ATR stop level.",
            rationale="The 15m pullback rejected EMA and session VWAP with RSI turning back down while the enabled 4h trend-strength filters remained bearish.",
            time_stop_bars=params.time_stop_bars,
        )
    return None


def build_trend_pullback_strategy_class():
    from backtesting import Strategy

    def _writable_array(series: pd.Series) -> np.ndarray:
        return np.array(series.to_numpy(copy=True), copy=True)

    class TrendPullbackStrategy(Strategy):
        ema_fast = 12
        ema_trend = 30
        ema_bias = 120
        atr_period = 14
        atr_stop_multiplier = 1.2
        reward_to_risk = 3.0
        time_stop_bars = 20
        higher_tf_validation = True
        allow_long = True
        allow_short = True
        use_4h_confirmation = True
        higher_tf_4h_fast = 20
        higher_tf_4h_slow = 50
        higher_tf_adx_period = 14
        higher_tf_adx_threshold = 28.0
        higher_tf_slope_bars = 3
        rsi_period = 14
        rsi_trend_threshold = 51.0
        rsi_long_pullback_floor = 38.0
        rsi_long_pullback_ceiling = 50.0
        rsi_short_pullback_floor = 50.0
        rsi_short_pullback_ceiling = 62.0
        rsi_turn_threshold = 2.0
        pullback_atr_buffer = 0.25
        trigger_lookback = 1
        volume_window = 20
        volume_ratio_threshold = 1.0
        require_volume_confirmation = False
        require_vwap_confirmation = True
        trailing_stop_enabled = False
        trailing_stop_lookback = 10
        trailing_stop_atr_multiplier = 2.5

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
            params = TrendPullbackParams(
                ema_fast=self.ema_fast,
                ema_trend=self.ema_trend,
                ema_bias=self.ema_bias,
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
                higher_tf_slope_bars=self.higher_tf_slope_bars,
                rsi_period=self.rsi_period,
                rsi_trend_threshold=self.rsi_trend_threshold,
                rsi_long_pullback_floor=self.rsi_long_pullback_floor,
                rsi_long_pullback_ceiling=self.rsi_long_pullback_ceiling,
                rsi_short_pullback_floor=self.rsi_short_pullback_floor,
                rsi_short_pullback_ceiling=self.rsi_short_pullback_ceiling,
                rsi_turn_threshold=self.rsi_turn_threshold,
                pullback_atr_buffer=self.pullback_atr_buffer,
                trigger_lookback=self.trigger_lookback,
                volume_window=self.volume_window,
                volume_ratio_threshold=self.volume_ratio_threshold,
                require_volume_confirmation=self.require_volume_confirmation,
                require_vwap_confirmation=self.require_vwap_confirmation,
                trailing_stop_enabled=self.trailing_stop_enabled,
                trailing_stop_lookback=self.trailing_stop_lookback,
                trailing_stop_atr_multiplier=self.trailing_stop_atr_multiplier,
            )
            signal_frame = compute_trend_pullback_frame(data, params)
            self.signal = self.I(lambda: _writable_array(signal_frame["signal"]), plot=False)
            self.long_stop = self.I(lambda: _writable_array(signal_frame["long_stop"]), plot=False)
            self.short_stop = self.I(lambda: _writable_array(signal_frame["short_stop"]), plot=False)
            self.long_take_profit = self.I(
                lambda: _writable_array(signal_frame["long_take_profit"]), plot=False
            )
            self.short_take_profit = self.I(
                lambda: _writable_array(signal_frame["short_take_profit"]), plot=False
            )
            self.long_trailing_stop = self.I(
                lambda: _writable_array(signal_frame["long_trailing_stop"]), plot=False
            )
            self.short_trailing_stop = self.I(
                lambda: _writable_array(signal_frame["short_trailing_stop"]), plot=False
            )

        def next(self):
            if self.position and self.trades:
                active_trade = self.trades[-1]
                if self.trailing_stop_enabled:
                    if active_trade.is_long:
                        trailing_stop = float(self.long_trailing_stop[-1])
                        if np.isfinite(trailing_stop):
                            active_trade.sl = max(float(active_trade.sl or -np.inf), trailing_stop)
                    else:
                        trailing_stop = float(self.short_trailing_stop[-1])
                        if np.isfinite(trailing_stop):
                            active_trade.sl = min(float(active_trade.sl or np.inf), trailing_stop)
                if len(self.data) - 1 - active_trade.entry_bar >= self.time_stop_bars:
                    self.position.close()
                    return

            if self.position:
                return

            if self.signal[-1] == 1:
                self.buy(sl=float(self.long_stop[-1]), tp=float(self.long_take_profit[-1]))
            elif self.signal[-1] == -1:
                self.sell(sl=float(self.short_stop[-1]), tp=float(self.short_take_profit[-1]))

    TrendPullbackStrategy.__name__ = "TrendPullbackStrategy"
    return TrendPullbackStrategy
