"""Monetary objective for comparing runs: lower total_score is better."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite

from analysis.metrics import Metrics
from simulation.engine import SimulationResult


@dataclass(frozen=True)
class Score:
    currency: str
    electricity_cost: float
    cycle_cost: float
    comfort_cost: float
    total_score: float


def nonnegative(value, name: str) -> float:
    number = float(value)
    if not isfinite(number) or number < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return number


def minute_of_day(value: str) -> int:
    if not isinstance(value, str) or len(value) != 5:
        raise ValueError('Tariff times must be quoted HH:MM strings')
    parsed = datetime.strptime(value, '%H:%M')
    return parsed.hour * 60 + parsed.minute


@dataclass(frozen=True)
class ScoringParams:
    currency: str
    cost_per_start: float
    cost_per_comfort_degree_hour: float
    minute_rates: tuple[float, ...]
    boundaries: tuple[int, ...]

    @classmethod
    def from_config(cls, config: dict):
        currency = config['currency']
        if not isinstance(currency, str) or not currency.strip():
            raise ValueError('Scoring currency must be a nonempty label')
        electricity = config['electricity']
        windows_enabled = electricity.get('tariff_windows_enabled', True)
        if not isinstance(windows_enabled, bool):
            raise ValueError('tariff_windows_enabled must be a boolean')
        default = nonnegative(electricity['default_price_per_kWh'], 'default_price_per_kWh')
        rates = [default] * 1440
        covered = set()
        boundaries = {0, 1440}
        for window in (electricity.get('tariff_windows', []) if windows_enabled else []):
            start, end = minute_of_day(window['start']), minute_of_day(window['end'])
            if start == end:
                raise ValueError('Tariff window start and end must differ; use the default for all day')
            price = nonnegative(window['price_per_kWh'], 'price_per_kWh')
            minutes = (list(range(start, end)) if start < end
                       else list(range(start, 1440)) + list(range(end)))
            if covered.intersection(minutes):
                raise ValueError('Electricity tariff windows must not overlap')
            covered.update(minutes)
            for minute in minutes:
                rates[minute] = price
            boundaries.update((start, end))
        return cls(currency, nonnegative(config['cost_per_heat_pump_start'], 'cost_per_heat_pump_start'),
                   nonnegative(config['cost_per_comfort_degree_hour'], 'cost_per_comfort_degree_hour'),
                   tuple(rates), tuple(sorted(boundaries)))

    def step_cost(self, timestamp: str, duration: timedelta, power_W: float) -> float:
        """Electricity cost of one step at the tariff of each minute."""
        current = datetime.fromisoformat(timestamp)
        if current.tzinfo is not None:
            raise ValueError('Tariffs require local simulation timestamps without timezone offsets')
        end = current + duration
        cost = 0.0
        while current < end:
            midnight = current.replace(hour=0, minute=0, second=0, microsecond=0)
            next_boundary = next(midnight + timedelta(minutes=minute)
                                 for minute in self.boundaries
                                 if midnight + timedelta(minutes=minute) > current)
            stop = min(end, next_boundary)
            rate = self.minute_rates[current.hour * 60 + current.minute]
            cost += power_W * rate / 1000 * (stop - current).total_seconds() / 3600
            current = stop
        return cost

    def transition_score(self, timestamp: datetime, dt_hours: float, power_W: float,
                         was_on: bool, is_on: bool, room_C: float,
                         comfort_threshold_C: float, *, start_count: int | None = None) -> Score:
        """Incremental training cost in the same units as the aggregate score."""
        electricity = self.step_cost(timestamp.isoformat(), timedelta(hours=dt_hours), power_W)
        cycle = self.start_cost(int(is_on and not was_on) if start_count is None else start_count)
        comfort = self.comfort_cost(max(0.0, comfort_threshold_C - room_C) * dt_hours)
        return Score(self.currency, electricity, cycle, comfort, electricity + cycle + comfort)

    def start_cost(self, starts: int) -> float:
        return starts * self.cost_per_start

    def comfort_cost(self, degree_hours: float) -> float:
        return degree_hours * self.cost_per_comfort_degree_hour


def calculate_score(result: SimulationResult, metrics: Metrics,
                    dt_hours: float, params: ScoringParams) -> Score:
    dt_hours = nonnegative(dt_hours, 'dt_hours')
    duration = timedelta(hours=dt_hours)
    if duration <= timedelta(0):
        raise ValueError('dt_hours must be positive')
    if len(result.timestamps) != len(result.electric_power_W):
        raise ValueError('Timestamps and electric powers must have equal lengths')
    electricity_cost = sum(params.step_cost(timestamp, duration, nonnegative(power, 'electric_power_W'))
                           for timestamp, power in zip(result.timestamps, result.electric_power_W))
    cycle_cost = params.start_cost(metrics.heat_pump_start_count)
    comfort_cost = params.comfort_cost(metrics.comfort_degree_hours)
    total = electricity_cost + cycle_cost + comfort_cost
    if not isfinite(total):
        raise ValueError('Total score must be finite')
    return Score(params.currency, electricity_cost, cycle_cost, comfort_cost, total)
