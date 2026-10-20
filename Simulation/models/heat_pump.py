from __future__ import annotations
from bisect import bisect_left
from dataclasses import dataclass
from math import isfinite
from models.performance_map import PerformanceMap


@dataclass(frozen=True)
class HeatPumpParams:
    source_temp_min_C: float
    sink_temp_max_C: float
    performance_map: PerformanceMap
    min_on_time_min: float = 0
    min_off_time_min: float = 0
    max_volume_flow_L_h: float | None = None
    minimum_supply_spread_K: float = 0.01
    min_volume_flow_L_h: float = 0.0
    supply_spread_K: float = 5.0

    def __post_init__(self):
        if not isfinite(self.supply_spread_K) or self.supply_spread_K <= 0:
            raise ValueError('WP supply_spread_K (WP supply-return spread) must be finite and positive')
        if not isfinite(self.min_volume_flow_L_h) or self.min_volume_flow_L_h < 0:
            raise ValueError('WP min_volume_flow_L_h must be finite and nonnegative')
        if (self.max_volume_flow_L_h is not None
                and self.min_volume_flow_L_h > self.max_volume_flow_L_h):
            raise ValueError('WP min_volume_flow_L_h must not exceed max_volume_flow_L_h')
        for name in ("min_on_time_min", "min_off_time_min"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"Heat pump {name} must be finite and nonnegative")
        if not isfinite(self.minimum_supply_spread_K) or self.minimum_supply_spread_K <= 0:
            raise ValueError('WP minimum_supply_spread_K must be finite and positive')
        if self.max_volume_flow_L_h is not None and (
                not isfinite(self.max_volume_flow_L_h) or self.max_volume_flow_L_h <= 0):
            raise ValueError('WP max_volume_flow_L_h must be positive or null')


def get_performance(source_temp_C: float, sink_temp_C: float,
                    params: HeatPumpParams, power_fraction: float = 1.0) -> tuple[float, float, float]:
    """Map command -1=off, 0=minimum, 1=maximum to manufacturer performance."""
    power_fraction = quantize_power_command(power_fraction, params)
    return params.performance_map.performance(
        max(source_temp_C, params.source_temp_min_C),
        min(sink_temp_C, params.sink_temp_max_C), power_fraction)


def validate_power_command(command):
    if not isfinite(command) or not (command == -1 or 0 <= command <= 1):
        raise ValueError('Power command must be -1 (off) or in [0, 1] (minimum to maximum)')


def available_power_fractions(params: HeatPumpParams) -> tuple[float, ...]:
    """Discrete commands mapped exactly to manufacturer inverter-speed columns."""
    return params.performance_map.power_fractions


def quantize_power_command(command, params: HeatPumpParams) -> float:
    validate_power_command(command)
    if command == -1:
        return -1.0
    levels = available_power_fractions(params)
    right = bisect_left(levels, command)
    if right == 0:
        return levels[0]
    if right == len(levels):
        return levels[-1]
    lower, upper = levels[right - 1], levels[right]
    # Nearest level; on a tie the lower one, exactly as min() over ascending levels.
    return lower if abs(lower - command) <= abs(upper - command) else upper


def heat_to_power_command(heat_W, source_C, supply_C, params):
    """Convert positive requested watts to modulation; zero demand means off."""
    if not isfinite(heat_W):
        raise ValueError('Requested heat must be finite')
    if heat_W <= 0:
        return -1.0
    levels = available_power_fractions(params)
    return next((level for level in levels
                 if get_performance(source_C, supply_C, params, level)[1] >= heat_W), levels[-1])
