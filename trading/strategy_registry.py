from __future__ import annotations

from collections.abc import Iterable

from trading.models import StrategyMetadata, StrategyParameterSpec
from trading.strategies.trend_pullback import (
    TrendPullbackParams,
    build_trend_pullback_strategy_class,
    compute_trend_pullback_frame,
    latest_trend_pullback_decision,
)
from trading.strategies.vwap_ema_cross import (
    VwapEmaCrossParams,
    build_vwap_ema_cross_strategy_class,
    compute_vwap_ema_cross_frame,
    latest_vwap_ema_cross_decision,
)


class StrategyDefinition:
    def __init__(
        self,
        metadata: StrategyMetadata,
        *,
        params_model,
        signal_frame_builder,
        latest_decision_builder,
        backtesting_strategy_builder,
    ):
        self.metadata = metadata
        self.params_model = params_model
        self.signal_frame_builder = signal_frame_builder
        self.latest_decision_builder = latest_decision_builder
        self.backtesting_strategy_builder = backtesting_strategy_builder

    def default_params(self) -> dict:
        return self.params_model().model_dump()

    def build_backtesting_strategy(self):
        return self.backtesting_strategy_builder()


class StrategyRegistry:
    def __init__(self, definitions: Iterable[StrategyDefinition]):
        self._definitions = {definition.metadata.id: definition for definition in definitions}

    def all(self) -> list[StrategyDefinition]:
        return list(self._definitions.values())

    def list_metadata(self) -> list[StrategyMetadata]:
        return [definition.metadata for definition in self.all()]

    def get(self, strategy_id: str) -> StrategyDefinition:
        try:
            return self._definitions[strategy_id]
        except KeyError as exc:
            raise KeyError(f"unsupported strategy: {strategy_id}") from exc


def _param_spec(name: str, default, description: str, *, minimum=None, maximum=None, step=None):
    if isinstance(default, bool):
        kind = "boolean"
    elif isinstance(default, float):
        kind = "number"
    else:
        kind = "integer"
    return StrategyParameterSpec(
        name=name,
        label=name.replace("_", " ").title(),
        type=kind,
        default=default,
        minimum=minimum,
        maximum=maximum,
        step=step,
        description=description,
    )


def build_default_registry() -> StrategyRegistry:
    trend_pullback_params = TrendPullbackParams()
    trend_pullback_definition = StrategyDefinition(
        metadata=StrategyMetadata(
            id="trend_pullback",
            name="Trend Pullback",
            description="Trades 15m BTC pullbacks only when the 4h regime is trending and the 15m chart reclaims session VWAP with an RSI turn back into trend.",
            supported_markets=["BTC-USD"],
            supported_timeframes=["15m"],
            default_params=trend_pullback_params.model_dump(),
            param_schema=[
                _param_spec("ema_fast", trend_pullback_params.ema_fast, "Pullback reference EMA.", minimum=2, maximum=100, step=1),
                _param_spec("ema_trend", trend_pullback_params.ema_trend, "Primary trend EMA.", minimum=3, maximum=200, step=1),
                _param_spec("ema_bias", trend_pullback_params.ema_bias, "Higher-timeframe bias EMA.", minimum=5, maximum=400, step=1),
                _param_spec("atr_period", trend_pullback_params.atr_period, "ATR period used for stop placement.", minimum=2, maximum=100, step=1),
                _param_spec(
                    "atr_stop_multiplier",
                    trend_pullback_params.atr_stop_multiplier,
                    "ATR multiple used when comparing stop distance.",
                    minimum=0.5,
                    maximum=5.0,
                    step=0.1,
                ),
                _param_spec(
                    "reward_to_risk",
                    trend_pullback_params.reward_to_risk,
                    "Take-profit multiple of initial risk.",
                    minimum=0.5,
                    maximum=5.0,
                    step=0.1,
                ),
                _param_spec(
                    "time_stop_bars",
                    trend_pullback_params.time_stop_bars,
                    "Maximum holding period in bars before force-closing.",
                    minimum=1,
                    maximum=100,
                    step=1,
                ),
                _param_spec(
                    "higher_tf_validation",
                    trend_pullback_params.higher_tf_validation,
                    "Require 4h trend alignment before allowing 15m entries.",
                ),
                _param_spec(
                    "use_4h_confirmation",
                    trend_pullback_params.use_4h_confirmation,
                    "Use 4h trend and ADX/DMI confirmation in the higher-timeframe filter.",
                ),
                _param_spec(
                    "allow_long",
                    trend_pullback_params.allow_long,
                    "Enable long trades when the broader market regime supports them.",
                ),
                _param_spec(
                    "allow_short",
                    trend_pullback_params.allow_short,
                    "Enable short trades when the broader market regime supports them.",
                ),
                _param_spec(
                    "higher_tf_4h_fast",
                    trend_pullback_params.higher_tf_4h_fast,
                    "Fast EMA used on the 4h validation trend.",
                    minimum=2,
                    maximum=200,
                    step=1,
                ),
                _param_spec(
                    "higher_tf_4h_slow",
                    trend_pullback_params.higher_tf_4h_slow,
                    "Slow EMA used on the 4h validation trend.",
                    minimum=3,
                    maximum=300,
                    step=1,
                ),
                _param_spec(
                    "higher_tf_adx_period",
                    trend_pullback_params.higher_tf_adx_period,
                    "ADX/DMI period used on the enabled higher timeframes.",
                    minimum=2,
                    maximum=100,
                    step=1,
                ),
                _param_spec(
                    "higher_tf_adx_threshold",
                    trend_pullback_params.higher_tf_adx_threshold,
                    "Minimum 4h ADX required before trend-pullback entries are allowed.",
                    minimum=0.0,
                    maximum=100.0,
                    step=1.0,
                ),
                _param_spec(
                    "higher_tf_slope_bars",
                    trend_pullback_params.higher_tf_slope_bars,
                    "Number of 4h bars used to confirm the slow EMA is still sloping with trend.",
                    minimum=1,
                    maximum=50,
                    step=1,
                ),
                _param_spec(
                    "rsi_period",
                    trend_pullback_params.rsi_period,
                    "RSI period used on the 15m trigger chart.",
                    minimum=2,
                    maximum=100,
                    step=1,
                ),
                _param_spec(
                    "rsi_trend_threshold",
                    trend_pullback_params.rsi_trend_threshold,
                    "RSI level the trigger bar must reclaim to confirm trend resumption.",
                    minimum=0.0,
                    maximum=100.0,
                    step=1.0,
                ),
                _param_spec(
                    "rsi_long_pullback_floor",
                    trend_pullback_params.rsi_long_pullback_floor,
                    "Lower bound of the acceptable long pullback RSI zone on the prior bar.",
                    minimum=0.0,
                    maximum=100.0,
                    step=1.0,
                ),
                _param_spec(
                    "rsi_long_pullback_ceiling",
                    trend_pullback_params.rsi_long_pullback_ceiling,
                    "Upper bound of the acceptable long pullback RSI zone on the prior bar.",
                    minimum=0.0,
                    maximum=100.0,
                    step=1.0,
                ),
                _param_spec(
                    "rsi_short_pullback_floor",
                    trend_pullback_params.rsi_short_pullback_floor,
                    "Lower bound of the acceptable short pullback RSI zone on the prior bar.",
                    minimum=0.0,
                    maximum=100.0,
                    step=1.0,
                ),
                _param_spec(
                    "rsi_short_pullback_ceiling",
                    trend_pullback_params.rsi_short_pullback_ceiling,
                    "Upper bound of the acceptable short pullback RSI zone on the prior bar.",
                    minimum=0.0,
                    maximum=100.0,
                    step=1.0,
                ),
                _param_spec(
                    "rsi_turn_threshold",
                    trend_pullback_params.rsi_turn_threshold,
                    "Minimum one-bar RSI turn required on the trigger bar.",
                    minimum=0.0,
                    maximum=50.0,
                    step=0.5,
                ),
                _param_spec(
                    "pullback_atr_buffer",
                    trend_pullback_params.pullback_atr_buffer,
                    "ATR buffer used around the EMA/session-VWAP pullback zone.",
                    minimum=0.0,
                    maximum=5.0,
                    step=0.05,
                ),
                _param_spec(
                    "trigger_lookback",
                    trend_pullback_params.trigger_lookback,
                    "Number of prior bars the trigger bar must break to confirm resumption.",
                    minimum=1,
                    maximum=20,
                    step=1,
                ),
                _param_spec(
                    "volume_window",
                    trend_pullback_params.volume_window,
                    "Rolling volume window used to validate 15m participation on the resumption bar.",
                    minimum=2,
                    maximum=200,
                    step=1,
                ),
                _param_spec(
                    "volume_ratio_threshold",
                    trend_pullback_params.volume_ratio_threshold,
                    "Minimum ratio versus rolling median volume required on the trigger bar.",
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1,
                ),
                _param_spec(
                    "require_volume_confirmation",
                    trend_pullback_params.require_volume_confirmation,
                    "Require the trigger bar to clear the relative-volume threshold.",
                ),
                _param_spec(
                    "require_vwap_confirmation",
                    trend_pullback_params.require_vwap_confirmation,
                    "Require the trigger bar to close on the correct side of session VWAP.",
                ),
                _param_spec(
                    "trailing_stop_enabled",
                    trend_pullback_params.trailing_stop_enabled,
                    "Enable ATR-based trailing stop management after entry.",
                ),
                _param_spec(
                    "trailing_stop_lookback",
                    trend_pullback_params.trailing_stop_lookback,
                    "Lookback used for the chandelier-style trailing stop.",
                    minimum=2,
                    maximum=200,
                    step=1,
                ),
                _param_spec(
                    "trailing_stop_atr_multiplier",
                    trend_pullback_params.trailing_stop_atr_multiplier,
                    "ATR multiple used for the trailing stop.",
                    minimum=0.1,
                    maximum=20.0,
                    step=0.1,
                ),
            ],
        ),
        params_model=TrendPullbackParams,
        signal_frame_builder=compute_trend_pullback_frame,
        latest_decision_builder=latest_trend_pullback_decision,
        backtesting_strategy_builder=build_trend_pullback_strategy_class,
    )
    vwap_params = VwapEmaCrossParams()
    vwap_definition = StrategyDefinition(
        metadata=StrategyMetadata(
            id="vwap_ema_cross",
            name="VWAP EMA Cross",
            description="Trades 15m BTC 9/21 EMA crosses only when price is on the correct side of session VWAP and the 4h trend-strength filter is aligned.",
            supported_markets=["BTC-USD"],
            supported_timeframes=["15m"],
            default_params=vwap_params.model_dump(),
            param_schema=[
                _param_spec("ema_fast", vwap_params.ema_fast, "Fast EMA used for the 15m crossover trigger.", minimum=2, maximum=100, step=1),
                _param_spec("ema_slow", vwap_params.ema_slow, "Slow EMA used for the 15m crossover trigger.", minimum=3, maximum=200, step=1),
                _param_spec("atr_period", vwap_params.atr_period, "ATR period used for stop placement.", minimum=2, maximum=100, step=1),
                _param_spec("atr_stop_multiplier", vwap_params.atr_stop_multiplier, "ATR multiple used for the initial stop.", minimum=0.5, maximum=5.0, step=0.1),
                _param_spec("reward_to_risk", vwap_params.reward_to_risk, "Take-profit multiple of initial risk.", minimum=0.5, maximum=5.0, step=0.1),
                _param_spec("time_stop_bars", vwap_params.time_stop_bars, "Maximum holding period in bars before force-closing.", minimum=1, maximum=100, step=1),
                _param_spec("higher_tf_validation", vwap_params.higher_tf_validation, "Require 4h regime alignment before allowing 15m entries."),
                _param_spec("use_4h_confirmation", vwap_params.use_4h_confirmation, "Use 4h EMA and ADX/DMI confirmation in the higher-timeframe filter."),
                _param_spec("allow_long", vwap_params.allow_long, "Enable long trades when the broader market regime supports them."),
                _param_spec("allow_short", vwap_params.allow_short, "Enable short trades when the broader market regime supports them."),
                _param_spec("higher_tf_4h_fast", vwap_params.higher_tf_4h_fast, "Fast EMA used on the 4h validation trend.", minimum=2, maximum=200, step=1),
                _param_spec("higher_tf_4h_slow", vwap_params.higher_tf_4h_slow, "Slow EMA used on the 4h validation trend.", minimum=3, maximum=400, step=1),
                _param_spec("higher_tf_adx_period", vwap_params.higher_tf_adx_period, "ADX/DMI period used on the 4h timeframe.", minimum=2, maximum=100, step=1),
                _param_spec("higher_tf_adx_threshold", vwap_params.higher_tf_adx_threshold, "Minimum 4h ADX required when the ADX filter is enabled.", minimum=0.0, maximum=100.0, step=1.0),
                _param_spec("require_htf_adx", vwap_params.require_htf_adx, "Require 4h ADX strength in addition to EMA alignment."),
                _param_spec("rsi_period", vwap_params.rsi_period, "RSI period used for basic momentum confirmation on the 15m chart.", minimum=2, maximum=100, step=1),
                _param_spec("rsi_threshold", vwap_params.rsi_threshold, "RSI threshold required for long state; shorts use the inverse threshold.", minimum=0.0, maximum=100.0, step=1.0),
                _param_spec("swing_lookback", vwap_params.swing_lookback, "Lookback used for swing-based stop placement.", minimum=1, maximum=50, step=1),
                _param_spec("break_even_enabled", vwap_params.break_even_enabled, "Move the stop to entry once the trade reaches the configured reward multiple."),
                _param_spec("break_even_reward_multiple", vwap_params.break_even_reward_multiple, "Reward multiple that activates the break-even stop.", minimum=0.1, maximum=10.0, step=0.1),
            ],
        ),
        params_model=VwapEmaCrossParams,
        signal_frame_builder=compute_vwap_ema_cross_frame,
        latest_decision_builder=latest_vwap_ema_cross_decision,
        backtesting_strategy_builder=build_vwap_ema_cross_strategy_class,
    )
    return StrategyRegistry([trend_pullback_definition, vwap_definition])
