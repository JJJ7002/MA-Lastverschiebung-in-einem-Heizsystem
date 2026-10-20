from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time

from control.temperature_targets import HeatingCurveParams, NightSetback
from models.heat_pump import HeatPumpParams


@dataclass(frozen=True)
class ScheduleWindow:
    start: time
    end: time
    mode: str

    def __post_init__(self) -> None:
        if self.mode not in ('force_on', 'force_off'):
            raise ValueError('Schedule mode must be force_on or force_off')
        if self.start == self.end:
            raise ValueError('Schedule start and end must differ')

    def contains(self, current: time) -> bool:
        if self.start < self.end:
            return self.start <= current < self.end
        return current >= self.start or current < self.end


@dataclass(frozen=True)
class ScheduleOverride:
    enabled: bool = False
    windows: tuple[ScheduleWindow, ...] = ()
    seasonal_heating_off_enabled: bool = False
    seasonal_heating_off_start: tuple[int, int] = (6, 1)
    seasonal_heating_off_end: tuple[int, int] = (9, 1)

    @classmethod
    def from_config(cls, config: dict) -> ScheduleOverride:
        enabled = config.get('enabled', False)
        if not isinstance(enabled, bool):
            raise ValueError('Schedule enabled must be a boolean')
        seasonal = config.get('seasonal_heating_off', {})
        seasonal_enabled = seasonal.get('enabled', False)
        if not isinstance(seasonal_enabled, bool):
            raise ValueError('Schedule seasonal_heating_off.enabled must be a boolean')

        def parse_month_day(value: str, name: str) -> tuple[int, int]:
            if not isinstance(value, str) or len(value) != 5 or value[2] != '-':
                raise ValueError(f'Schedule {name} must use quoted MM-DD format')
            try:
                parsed = datetime.strptime(f'2000-{value}', '%Y-%m-%d')
            except ValueError as error:
                raise ValueError(f'Schedule {name} must be a valid calendar date in MM-DD format') from error
            return parsed.month, parsed.day

        seasonal_start = parse_month_day(seasonal.get('start', '06-01'),
                                         'seasonal_heating_off.start')
        seasonal_end = parse_month_day(seasonal.get('end', '09-01'),
                                       'seasonal_heating_off.end')
        if seasonal_start == seasonal_end:
            raise ValueError('Schedule seasonal_heating_off.start and end must differ')
        windows = []
        for window in config.get('windows', []):
            def parse_time(value: str) -> time:
                if not isinstance(value, str) or len(value) != 5:
                    raise ValueError('Schedule times must be quoted HH:MM strings')
                return datetime.strptime(value, '%H:%M').time()
            windows.append(ScheduleWindow(parse_time(window['start']),
                                          parse_time(window['end']), window['mode']))
        if enabled and not windows:
            raise ValueError('Enabled schedule requires at least one window')
        return cls(enabled, tuple(windows), seasonal_enabled, seasonal_start, seasonal_end)

    def heating_off_at(self, timestamp: datetime | None) -> bool:
        if not self.seasonal_heating_off_enabled:
            return False
        if timestamp is None:
            raise ValueError('Seasonal heating-off schedule requires a simulation timestamp')
        current = (timestamp.month, timestamp.day)
        start, end = self.seasonal_heating_off_start, self.seasonal_heating_off_end
        return start <= current < end if start < end else current >= start or current < end

    def mode_at(self, timestamp: datetime | None) -> str | None:
        seasonal_off = self.heating_off_at(timestamp)
        if not self.enabled and not seasonal_off:
            return None
        if timestamp is None:
            raise ValueError('Enabled schedule requires a simulation timestamp')
        modes = ({window.mode for window in self.windows if window.contains(timestamp.time())}
                 if self.enabled else set())
        if seasonal_off:
            modes.add('force_off')
        # A blocking window always wins over a simultaneous forced-on window.
        if 'force_off' in modes:
            return 'force_off'
        return 'force_on' if 'force_on' in modes else None


@dataclass(frozen=True)
class OperatingParams:
    heating_curve: HeatingCurveParams
    schedule: ScheduleOverride = ScheduleOverride()
    night_setback: NightSetback = NightSetback()



@dataclass
class OperatingState:
    is_on: bool = False
    minutes_in_current_state: int = 0
