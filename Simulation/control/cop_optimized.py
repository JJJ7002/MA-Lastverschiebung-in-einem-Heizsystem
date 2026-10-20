"""COP feedback control using a fixed reference for each calendar day."""
from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite

from control.operating_constraints import OperatingParams, OperatingState
from models.heat_pump import (HeatPumpParams, get_performance, quantize_power_command,
                              validate_power_command)


@dataclass(frozen=True)
class CopOptimizedParams:
    daily_reference_cop: dict[date, float]
    full_power_advantage: float = 0.2
    min_power_advantage: float = 0.0

    def __post_init__(self):
        if (not isfinite(self.min_power_advantage) or not isfinite(self.full_power_advantage)
                or not -1 < self.min_power_advantage < self.full_power_advantage):
            raise ValueError('COP thresholds must be finite with -1 < min_power_advantage < full_power_advantage')



@dataclass
class CopControlState(OperatingState):
    fallback_active: bool = False
    fallback_running: bool = False
    fallback_override: bool = False


def power_from_advantage(advantage: float, full_advantage: float,
                         min_advantage: float = 0.0) -> float:
    if advantage < min_advantage:
        return -1.0
    return min(1.0, (advantage - min_advantage) / (full_advantage - min_advantage))


def update_cop_control(storage_C: float, outside_C: float, timestamp: datetime,
                       state: OperatingState, control: OperatingParams,
                       heat_pump: HeatPumpParams, params: CopOptimizedParams,
                       step_minutes: int, *, fallback_power: float = -1.0,
                       supply_target_C: float | None = None) -> tuple[OperatingState, float, float, float]:
    reference = params.daily_reference_cop[timestamp.date()]
    current, _, _ = get_performance(outside_C, storage_C if supply_target_C is None else supply_target_C, heat_pump)
    if not isfinite(current) or current <= 0 or not isfinite(reference) or reference <= 0:
        raise ValueError('COP control requires finite positive current and reference COP')
    advantage = current / reference - 1
    requested = power_from_advantage(advantage, params.full_power_advantage,
                                     params.min_power_advantage)
    validate_power_command(fallback_power)
    cop_requested = requested
    requested = max(requested, fallback_power)
    mode = control.schedule.mode_at(timestamp)
    if mode == 'force_on':
        requested = 1.0
    elapsed = state.minutes_in_current_state + step_minutes
    # Temperature ceiling and scheduled blocking override minimum run time.
    if mode == 'force_off' or storage_C >= heat_pump.sink_temp_max_C:
        requested = -1.0
        on = False
    elif state.is_on:
        on = requested >= 0 or elapsed < heat_pump.min_on_time_min
        if on:
            requested = max(0.0, requested)
    else:
        on = requested >= 0 and elapsed >= heat_pump.min_off_time_min
    updated = CopControlState(on, elapsed if on == state.is_on else 0, fallback_power >= 0 and on)
    # Scheduled forced operation is not counted as temperature fallback.
    updated.fallback_running = on and fallback_power >= 0 and mode is None
    updated.fallback_override = updated.fallback_running and fallback_power > cop_requested
    command = max(0.0, requested) if on else -1.0
    return updated, quantize_power_command(command, heat_pump), reference, advantage
