from __future__ import annotations

from collections.abc import Iterable

from trading.models import StrategyMetadata, StrategyParameterSpec
from trading.strategies.trend_pullback import (
    TrendPullbackParams,
    build_trend_pullback_strategy_class,
    compute_trend_pullback_frame,
    latest_trend_pullback_decision,
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
    params = TrendPullbackParams()
    definition = StrategyDefinition(
        metadata=StrategyMetadata(
            id="trend_pullback",
            name="Trend Pullback",
            description="Trades 15m BTC pullbacks only when local continuation signals align with a 4h higher-timeframe trend and strength filter.",
            supported_markets=["BTC-USD"],
            supported_timeframes=["15m"],
            default_params=params.model_dump(),
            param_schema=[
                _param_spec("ema_fast", params.ema_fast, "Pullback reference EMA.", minimum=2, maximum=100, step=1),
                _param_spec("ema_trend", params.ema_trend, "Primary trend EMA.", minimum=3, maximum=200, step=1),
                _param_spec("ema_bias", params.ema_bias, "Higher-timeframe bias EMA.", minimum=5, maximum=400, step=1),
                _param_spec("atr_period", params.atr_period, "ATR period used for stop placement.", minimum=2, maximum=100, step=1),
                _param_spec(
                    "atr_stop_multiplier",
                    params.atr_stop_multiplier,
                    "ATR multiple used when comparing stop distance.",
                    minimum=0.5,
                    maximum=5.0,
                    step=0.1,
                ),
                _param_spec(
                    "reward_to_risk",
                    params.reward_to_risk,
                    "Take-profit multiple of initial risk.",
                    minimum=0.5,
                    maximum=5.0,
                    step=0.1,
                ),
                _param_spec(
                    "time_stop_bars",
                    params.time_stop_bars,
                    "Maximum holding period in bars before force-closing.",
                    minimum=1,
                    maximum=100,
                    step=1,
                ),
                _param_spec(
                    "higher_tf_validation",
                    params.higher_tf_validation,
                    "Require 4h trend alignment before allowing 15m entries.",
                ),
                _param_spec(
                    "use_4h_confirmation",
                    params.use_4h_confirmation,
                    "Use 4h trend and ADX/DMI confirmation in the higher-timeframe filter.",
                ),
                _param_spec(
                    "allow_long",
                    params.allow_long,
                    "Enable long trades when the broader market regime supports them.",
                ),
                _param_spec(
                    "allow_short",
                    params.allow_short,
                    "Enable short trades when the broader market regime supports them.",
                ),
                _param_spec(
                    "higher_tf_4h_fast",
                    params.higher_tf_4h_fast,
                    "Fast EMA used on the 4h validation trend.",
                    minimum=2,
                    maximum=200,
                    step=1,
                ),
                _param_spec(
                    "higher_tf_4h_slow",
                    params.higher_tf_4h_slow,
                    "Slow EMA used on the 4h validation trend.",
                    minimum=3,
                    maximum=300,
                    step=1,
                ),
                _param_spec(
                    "higher_tf_adx_period",
                    params.higher_tf_adx_period,
                    "ADX/DMI period used on the enabled higher timeframes.",
                    minimum=2,
                    maximum=100,
                    step=1,
                ),
                _param_spec(
                    "higher_tf_adx_threshold",
                    params.higher_tf_adx_threshold,
                    "Minimum ADX required on enabled higher timeframes.",
                    minimum=0.0,
                    maximum=100.0,
                    step=1.0,
                ),
                _param_spec(
                    "macd_fast",
                    params.macd_fast,
                    "Fast MACD EMA used for 15m resumption confirmation.",
                    minimum=2,
                    maximum=100,
                    step=1,
                ),
                _param_spec(
                    "macd_slow",
                    params.macd_slow,
                    "Slow MACD EMA used for 15m resumption confirmation.",
                    minimum=3,
                    maximum=200,
                    step=1,
                ),
                _param_spec(
                    "macd_signal",
                    params.macd_signal,
                    "MACD signal EMA used for 15m resumption confirmation.",
                    minimum=2,
                    maximum=100,
                    step=1,
                ),
                _param_spec(
                    "volume_window",
                    params.volume_window,
                    "Rolling volume window used to validate 15m participation on the resumption bar.",
                    minimum=2,
                    maximum=200,
                    step=1,
                ),
                _param_spec(
                    "volume_ratio_threshold",
                    params.volume_ratio_threshold,
                    "Minimum ratio versus rolling median volume required on the trigger bar.",
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1,
                ),
                _param_spec(
                    "vwap_window",
                    params.vwap_window,
                    "Rolling VWAP window used to confirm the 15m trigger bar is back on the correct side of value.",
                    minimum=2,
                    maximum=200,
                    step=1,
                ),
                _param_spec(
                    "trailing_stop_enabled",
                    params.trailing_stop_enabled,
                    "Enable ATR-based trailing stop management after entry.",
                ),
                _param_spec(
                    "trailing_stop_lookback",
                    params.trailing_stop_lookback,
                    "Lookback used for the chandelier-style trailing stop.",
                    minimum=2,
                    maximum=200,
                    step=1,
                ),
                _param_spec(
                    "trailing_stop_atr_multiplier",
                    params.trailing_stop_atr_multiplier,
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
    return StrategyRegistry([definition])
