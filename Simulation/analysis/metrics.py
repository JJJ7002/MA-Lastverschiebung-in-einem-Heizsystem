from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from datetime import datetime
from math import isfinite

from simulation.engine import SimulationResult


@dataclass(frozen=True)
class Metrics:
    total_electric_energy_Wh: float
    total_thermal_energy_Wh: float
    seasonal_cop: float
    average_room_temperature_C: float
    minimum_room_temperature_C: float
    maximum_room_temperature_C: float
    heat_pump_start_count: int
    heat_pump_starts_per_day: float
    heat_pump_on_share: float
    comfort_below_hours: float
    comfort_degree_hours: float
    comfort_threshold_C: float
    comfort_buffer_K: float
    fallback_active_hours: float
    fallback_override_hours: float
    final_air_temperature_C: float = 0.0
    final_mass_temperature_C: float | None = None
    building_energy_change_Wh: float = 0.0
    storage_energy_change_Wh: float = 0.0


def metric_report_fields() -> list[str]:
    hidden = {'average_room_temperature_C', 'minimum_room_temperature_C',
              'maximum_room_temperature_C', 'heat_pump_on_share',
              'comfort_threshold_C', 'comfort_buffer_K', 'fallback_active_hours'}
    return [field.name for field in fields(Metrics) if field.name not in hidden]


def metric_report(metrics: Metrics) -> dict:
    values = asdict(metrics)
    return {name: values[name] for name in metric_report_fields()}


def _count_starts(states: list[bool]) -> int:
    count = 0
    previous = False
    for value in states:
        if value and not previous:
            count += 1
        previous = value
    return count


def calculate_metrics(result: SimulationResult, comfort_buffer_K: float, dt_hours: float,
                      target_temperature_C: float) -> Metrics:
    if not isfinite(comfort_buffer_K) or comfort_buffer_K < 0:
        raise ValueError('comfort_buffer_K must be finite and nonnegative')
    comfort_threshold_C = target_temperature_C - comfort_buffer_K
    room_temps = result.room_temperatures_C
    if result.night_setback.enabled:
        if len(result.timestamps) != len(room_temps):
            raise ValueError('Timestamps and room temperatures must have equal lengths')
        deficits = [
            max(0.0, comfort_threshold_C - result.night_setback.room_reduction_K(
                datetime.fromisoformat(timestamp), comfort=True) - temp)
            for temp, timestamp in zip(room_temps, result.timestamps)
        ]
    else:
        deficits = [max(0.0, comfort_threshold_C - temp) for temp in room_temps]
    below_count = sum(deficit > 0 for deficit in deficits)
    total_steps = len(room_temps)
    starts = sum(result.heat_pump_starts) if result.heat_pump_starts else _count_starts(result.heat_pump_on)
    runtime = result.heat_pump_runtime_fractions or [float(on) for on in result.heat_pump_on]
    simulated_days = len(result.heat_pump_on) * dt_hours / 24
    return Metrics(
        final_air_temperature_C=room_temps[-1] if room_temps else 0.,
        final_mass_temperature_C=result.mass_temperatures_C[-1] if result.mass_temperatures_C else None,
        building_energy_change_Wh=sum(r.get('building_energy_change_Wh', 0.) for r in result.hydraulics),
        storage_energy_change_Wh=(result.storage_energy_change_Wh
                                  if getattr(result, 'storage_energy_change_Wh', None) is not None
                                  else result.hydraulics[-1].get('storage_energy_Wh', result.initial_storage_energy_Wh)
                                  - result.initial_storage_energy_Wh if result.hydraulics else 0.),
        total_electric_energy_Wh=result.total_electric_energy_Wh,
        total_thermal_energy_Wh=result.total_thermal_energy_Wh,
        seasonal_cop=result.seasonal_cop,
        average_room_temperature_C=sum(room_temps) / total_steps if total_steps else 0.0,
        minimum_room_temperature_C=min(room_temps) if room_temps else 0.0,
        maximum_room_temperature_C=max(room_temps) if room_temps else 0.0,
        heat_pump_start_count=starts,
        heat_pump_starts_per_day=starts / simulated_days if simulated_days > 0 else 0.0,
        heat_pump_on_share=sum(runtime) / total_steps if total_steps else 0.0,
        comfort_below_hours=below_count * dt_hours,
        comfort_degree_hours=sum(deficits) * dt_hours,
        comfort_threshold_C=comfort_threshold_C,
        comfort_buffer_K=comfort_buffer_K,
        fallback_active_hours=sum(active * weight for active, weight in zip(result.fallback_active,
            result.heat_pump_runtime_fractions or [1.0] * len(result.fallback_active))) * dt_hours,
        fallback_override_hours=sum(active * weight for active, weight in zip(result.fallback_override,
            result.heat_pump_runtime_fractions or [1.0] * len(result.fallback_override))) * dt_hours,
    )
