"""Score of a short trial window, counted exactly like the total score of a run.

Pause and run load shifting judge every trial by the difference
between the score of the unchanged reference and the score of the trial over
the same steps, from the start of the trial until both have recovered:
electricity cost (tariff of each step), heat pump start cost and comfort cost
(kelvin hours below the comfort threshold). Lower is better.
"""
from dataclasses import dataclass
from datetime import timedelta


@dataclass(frozen=True)
class WindowScore:
    electricity: float = 0.0
    starts: float = 0.0
    comfort: float = 0.0
    electric_energy_Wh: float = 0.0
    start_count: int = 0
    comfort_deficit_Kh: float = 0.0

    @property
    def total(self):
        return self.electricity + self.starts + self.comfort

    def __add__(self, other):
        return WindowScore(self.electricity + other.electricity, self.starts + other.starts,
                           self.comfort + other.comfort, self.electric_energy_Wh + other.electric_energy_Wh,
                           self.start_count + other.start_count,
                           self.comfort_deficit_Kh + other.comfort_deficit_Kh)


@dataclass(frozen=True)
class TrialScorer:
    scoring: object  # analysis.scoring.ScoringParams
    comfort_buffer_K: float = 0.0

    @classmethod
    def electricity_only(cls, price_per_kWh=1.0):
        """Fallback when no scoring is supplied: electricity at a flat price, no start or comfort cost."""
        from analysis.scoring import ScoringParams
        return cls(ScoringParams.from_config(dict(
            currency='CHF', cost_per_heat_pump_start=0.0, cost_per_comfort_degree_hour=0.0,
            electricity=dict(default_price_per_kWh=price_per_kWh, tariff_windows_enabled=False))))

    def step(self, params, timestamp, result):
        dt_hours = params.dt_hours
        electricity = self.scoring.step_cost(timestamp.isoformat(), timedelta(hours=dt_hours),
                                             result.electric_power_W)
        threshold = (params.target_room_temperature_C - self.comfort_buffer_K
                     - params.operating.night_setback.room_reduction_K(timestamp, comfort=True))
        deficit = max(0.0, threshold - result.state.building.T_room_C) * dt_hours
        return WindowScore(electricity, self.scoring.start_cost(result.heat_pump_start_count),
                           self.scoring.comfort_cost(deficit), result.electric_power_W * dt_hours,
                           result.heat_pump_start_count, deficit)

    def window(self, params, weather, first_step, results):
        """Summed score of consecutive step results beginning at weather index `first_step`."""
        total = WindowScore()
        for offset, result in enumerate(results):
            total = total + self.step(params, weather[first_step + offset].timestamp, result)
        return total


def trial_scorer(options):
    """The scorer injected by the engine, or the electricity-only fallback."""
    return options.get('scorer') or TrialScorer.electricity_only()


def gain_fields(reference, candidate):
    """Trial result fields: positive gains mean the trial scores better than the reference."""
    return dict(score_gain=reference.total - candidate.total,
                electricity_gain=reference.electricity - candidate.electricity,
                start_gain=reference.starts - candidate.starts,
                comfort_gain=reference.comfort - candidate.comfort,
                saved_electric_energy_Wh=reference.electric_energy_Wh - candidate.electric_energy_Wh,
                extra_starts=candidate.start_count - reference.start_count,
                comfort_deficit_change_Kh=candidate.comfort_deficit_Kh - reference.comfort_deficit_Kh)


GAIN_KEYS = ('score_gain', 'electricity_gain', 'start_gain', 'comfort_gain',
             'saved_electric_energy_Wh', 'extra_starts', 'comfort_deficit_change_Kh')


def summarize(accepted):
    """Sums of the gain fields over accepted trials."""
    return {key: sum(t[key] for t in accepted) for key in GAIN_KEYS}


REMOVED_OPTIONS = {'min_saving_Wh': 'min_score_gain', 'start_penalty_Wh': 'the scoring section',
                   'comfort_penalty_Wh_per_Kh': 'the scoring section', 'comfort_buffer_K': 'building.comfort_buffer_K',
                   'credit_residual_energy': None}


def reject_removed_options(method, options):
    """Trials are scored with the run's score now; fail loudly on options of the old Wh criterion."""
    for name, replacement in REMOVED_OPTIONS.items():
        if name in options:
            hint = f'; use {replacement}' if replacement else ''
            raise ValueError(f'{method}.{name} was removed: trials are rated by the score of the '
                             f'comparison window{hint}')
