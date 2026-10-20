from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from math import isfinite


@dataclass(frozen=True)
class RadiatorHeatingCurve:
    """Adjust the weather-dependent slope and vertical offset of a radiator curve."""
    radiator: object
    ua_W_K: float
    room_target_C: float
    slope_factor: float = 1.0
    offset_K: float = 0.0
    reference_outside_C: float = 20.0

    def __post_init__(self):
        if not isfinite(self.slope_factor) or self.slope_factor <= 0:
            raise ValueError('control.heating_curve.slope_factor must be finite and positive')
        if not isfinite(self.offset_K):
            raise ValueError('control.heating_curve.offset_K must be finite')
        if not isfinite(self.reference_outside_C):
            raise ValueError('control.heating_curve.reference_outside_C must be finite')

    def target_temperature_C(self, outside_C, room_target_C=None):
        room = self.room_target_C if room_target_C is None else room_target_C
        # Scale the weather-dependent rise relative to the curve's room-temperature
        # equilibrium point. This is the same geometry as changing a real curve's
        # slope around its reference point; offset_K then shifts it in parallel.
        heat = max(0.0, self.ua_W_K * (room - outside_C))
        excess, _ = self.radiator.temperatures(heat)
        reference_excess, _ = self.radiator.temperatures(
            max(0.0, self.ua_W_K * (room - self.reference_outside_C)))
        return room + reference_excess + self.slope_factor * (excess-reference_excess) + self.offset_K


@dataclass(frozen=True)
class NightSetback:
    enabled: bool = False
    start: time = time(22)
    end: time = time(6)
    offset_K: float = 3.0
    room_offset_K: float = 2.0
    warmup_minutes_pro_K: float = 60.0
    warmup_rounding: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError('Night setback enabled must be a boolean')
        if self.start == self.end:
            raise ValueError('Night setback start and end must differ')
        if not isfinite(self.offset_K) or self.offset_K < 0:
            raise ValueError('Night setback offset_K must be finite and nonnegative')
        for name in ('room_offset_K', 'warmup_minutes_pro_K'):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f'Night setback {name} must be finite and nonnegative')
        if not isfinite(self.warmup_rounding) or not 0 <= self.warmup_rounding <= 1:
            raise ValueError('Night setback warmup_rounding must be finite and between 0 and 1')

    @classmethod
    def from_config(cls, config: dict) -> NightSetback:
        if 'warmup_minutes' in config:
            raise ValueError('Replace warmup_minutes with warmup_minutes_pro_K (minutes per kelvin of room_offset_K)')
        def parse_time(value: str) -> time:
            if not isinstance(value, str) or len(value) != 5:
                raise ValueError('Night setback times must be quoted HH:MM strings')
            return datetime.strptime(value, '%H:%M').time()
        return cls(
            enabled=config.get('enabled', False),
            start=parse_time(config.get('start', '22:00')),
            end=parse_time(config.get('end', '06:00')),
            offset_K=float(config.get('offset_K', 3.0)),
            room_offset_K=float(config.get('room_offset_K', 2.0)),
            warmup_minutes_pro_K=float(config.get('warmup_minutes_pro_K', 60.0)),
            warmup_rounding=float(config.get('warmup_rounding', 1.0)),
        )

    @property
    def warmup_minutes(self) -> float:
        return self.room_offset_K * self.warmup_minutes_pro_K

    def reduction_K(self, timestamp: datetime | None) -> float:
        return self.offset_K if self.is_active(timestamp) else 0.0

    def is_active(self, timestamp: datetime | None, *, comfort: bool = False) -> bool:
        if not self.enabled:
            return False
        if timestamp is None:
            raise ValueError('Enabled night setback requires a simulation timestamp')
        start_seconds = self.start.hour * 3600 + self.start.minute * 60
        end_seconds = self.end.hour * 3600 + self.end.minute * 60
        duration = (end_seconds - start_seconds) % 86400
        if not comfort:
            duration = max(0, duration - self.warmup_minutes * 60)
        current_seconds = (timestamp.hour * 3600 + timestamp.minute * 60
                           + timestamp.second + timestamp.microsecond / 1e6)
        return (current_seconds - start_seconds) % 86400 < duration

    def room_reduction_K(self, timestamp: datetime | None, *, comfort: bool = False) -> float:
        # Comfort stays at the night level until the scheduled end; only the
        # room control target follows the advance warmup curve.
        if not self.is_active(timestamp, comfort=True):
            return 0.0
        if comfort:
            return self.room_offset_K
        start_seconds = self.start.hour * 3600 + self.start.minute * 60
        end_seconds = self.end.hour * 3600 + self.end.minute * 60
        duration = (end_seconds - start_seconds) % 86400
        # Keep the full warmup duration even when its start precedes setback.
        # In that case the night begins partway through the target curve.
        ramp_seconds = self.warmup_minutes * 60
        if ramp_seconds == 0:
            return self.room_offset_K
        current_seconds = (timestamp.hour * 3600 + timestamp.minute * 60
                           + timestamp.second + timestamp.microsecond / 1e6)
        remaining = duration - (current_seconds - start_seconds) % 86400
        remaining_fraction = min(1.0, remaining / ramp_seconds)
        # Blend a linear ramp with quintic smootherstep.
        x = remaining_fraction
        smooth = x**3 * (10 - 15 * x + 6 * x**2)
        return self.room_offset_K * ((1 - self.warmup_rounding) * x
                                     + self.warmup_rounding * smooth)


@dataclass(frozen=True)
class HeatingCurveParams:
    """Linear curve: target [°C] = slope * outside temperature [°C] + offset [°C]."""

    slope: float
    offset_C: float

    def __post_init__(self) -> None:
        if not isfinite(self.slope) or not isfinite(self.offset_C):
            raise ValueError("Heating curve slope and offset must be finite.")

    def target_temperature_C(self, outside_temperature_C: float, room_target_C=None) -> float:
        return self.slope * outside_temperature_C + self.offset_C
