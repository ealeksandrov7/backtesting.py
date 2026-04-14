from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from trading.models import StrategyDecision, TradeAction


class TrendPullbackParams(BaseModel):
    ema_fast: int = Field(default=12, ge=2, le=200)
    ema_trend: int = Field(default=40, ge=3, le=300)
    ema_bias: int = Field(default=160, ge=5, le=500)
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_multiplier: float = Field(default=1.0, gt=0.1, le=10.0)
    reward_to_risk: float = Field(default=2.0, gt=0.1, le=10.0)
    time_stop_bars: int = Field(default=12, ge=1, le=200)
    higher_tf_validation: bool = True
    allow_long: bool = True
    allow_short: bool = True
    use_4h_confirmation: bool = True
    higher_tf_4h_fast: int = Field(default=20, ge=2, le=200)
    higher_tf_4h_slow: int = Field(default=50, ge=3, le=300)
    higher_tf_adx_period: int = Field(default=14, ge=2, le=100)
    higher_tf_adx_threshold: float = Field(default=15.0, ge=0.0, le=100.0)
    macd_fast: int = Field(default=12, ge=2, le=100)
    macd_slow: int = Field(default=26, ge=3, le=200)
    macd_signal: int = Field(default=9, ge=2, le=100)
    volume_window: int = Field(default=20, ge=2, le=200)
    volume_ratio_threshold: float = Field(default=0.8, gt=0.0, le=10.0)
    vwap_window: int = Field(default=20, ge=2, le=200)
    trailing_stop_enabled: bool = True
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


def _macd(series: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = _ema(series, fast)
    slow_ema = _ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _rolling_vwap(frame: pd.DataFrame, window: int) -> pd.Series:
    typical_price = (frame["High"] + frame["Low"] + frame["Close"]) / 3.0
    price_volume = typical_price * frame["Volume"]
    rolling_volume = frame["Volume"].rolling(window, min_periods=1).sum()
    return price_volume.rolling(window, min_periods=1).sum() / rolling_volume.replace(0, np.nan)


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
    frame["rolling_vwap"] = _rolling_vwap(frame, params.vwap_window)
    frame["macd"], frame["macd_signal"], frame["macd_hist"] = _macd(
        frame["Close"], params.macd_fast, params.macd_slow, params.macd_signal
    )
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
            frame["htf_4h_plus_di"] = plus_di_4h.reindex(frame.index, method="ffill")
            frame["htf_4h_minus_di"] = minus_di_4h.reindex(frame.index, method="ffill")
            frame["htf_4h_adx"] = adx_4h.reindex(frame.index, method="ffill")
            long_htf_regime &= (
                (frame["htf_4h_close"] > frame["htf_4h_fast"])
                & (frame["htf_4h_fast"] > frame["htf_4h_slow"])
                & (frame["htf_4h_plus_di"] > frame["htf_4h_minus_di"])
                & (frame["htf_4h_adx"] >= params.higher_tf_adx_threshold)
            ).fillna(False)
            short_htf_regime &= (
                (frame["htf_4h_close"] < frame["htf_4h_fast"])
                & (frame["htf_4h_fast"] < frame["htf_4h_slow"])
                & (frame["htf_4h_minus_di"] > frame["htf_4h_plus_di"])
                & (frame["htf_4h_adx"] >= params.higher_tf_adx_threshold)
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

    prev_long_pullback = (
        (frame["Low"].shift(1) <= frame["ema_fast"].shift(1))
        & (
            frame["Close"].shift(1).between(
                frame[["ema_fast", "ema_trend"]].min(axis=1).shift(1),
                frame[["ema_fast", "ema_trend"]].max(axis=1).shift(1),
            )
        )
    )
    prev_short_pullback = (
        (frame["High"].shift(1) >= frame["ema_fast"].shift(1))
        & (
            frame["Close"].shift(1).between(
                frame[["ema_fast", "ema_trend"]].min(axis=1).shift(1),
                frame[["ema_fast", "ema_trend"]].max(axis=1).shift(1),
            )
        )
    )

    bullish_confirmation = (
        (frame["Close"] > frame["Open"])
        & (frame["Close"] > frame["ema_fast"])
        & (frame["Close"] > frame["rolling_vwap"])
        & (frame["macd"] > frame["macd_signal"])
        & (frame["macd_hist"] > frame["macd_hist"].shift(1))
        & (frame["Volume"] >= frame["volume_median"] * params.volume_ratio_threshold)
    )
    bearish_confirmation = (
        (frame["Close"] < frame["Open"])
        & (frame["Close"] < frame["ema_fast"])
        & (frame["Close"] < frame["rolling_vwap"])
        & (frame["macd"] < frame["macd_signal"])
        & (frame["macd_hist"] < frame["macd_hist"].shift(1))
        & (frame["Volume"] >= frame["volume_median"] * params.volume_ratio_threshold)
    )

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
            rationale="The 15m pullback reclaimed local trend structure with MACD and volume confirmation while the enabled higher-timeframe filters remained bullish.",
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
            rationale="The 15m pullback reclaimed local downside structure with MACD and volume confirmation while the enabled higher-timeframe filters remained bearish.",
            time_stop_bars=params.time_stop_bars,
        )
    return None


def build_trend_pullback_strategy_class():
    from backtesting import Strategy

    def _writable_array(series: pd.Series) -> np.ndarray:
        return np.array(series.to_numpy(copy=True), copy=True)

    class TrendPullbackStrategy(Strategy):
        ema_fast = 12
        ema_trend = 40
        ema_bias = 160
        atr_period = 14
        atr_stop_multiplier = 1.0
        reward_to_risk = 2.0
        time_stop_bars = 12
        higher_tf_validation = True
        allow_long = True
        allow_short = True
        use_4h_confirmation = True
        higher_tf_4h_fast = 20
        higher_tf_4h_slow = 50
        higher_tf_adx_period = 14
        higher_tf_adx_threshold = 15.0
        macd_fast = 12
        macd_slow = 26
        macd_signal = 9
        volume_window = 20
        volume_ratio_threshold = 0.8
        vwap_window = 20
        trailing_stop_enabled = True
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
                macd_fast=self.macd_fast,
                macd_slow=self.macd_slow,
                macd_signal=self.macd_signal,
                volume_window=self.volume_window,
                volume_ratio_threshold=self.volume_ratio_threshold,
                vwap_window=self.vwap_window,
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
