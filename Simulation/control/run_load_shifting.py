"""Run load shifting: raised power within a running heat-pump cycle, judged by paired trials.

Only levels with a higher COP than the baseline level are tried (COP pre-filter);
acceptance depends on the score of the comparison window.

At a control interval where the baseline heat pump runs, the next higher map
level is tested for exactly this interval. The trial and the unchanged
reference are then simulated side by side under the heating curve, which
lowers the power again on its own, until room, building mass and storage have
returned to the reference. The trial is judged by the total score of the comparison window
(electricity, start and comfort cost, as in the run's score). Further levels are
tried one after another on the same interval while the net saving improves.
"""
from math import floor, isclose, isfinite

from control.pause_load_shifting import (DEFAULT_STORAGE_TOLERANCE_K, RECOVERY_MODES,
                                         _recovered, _recovered_mean)
from control.trial_scoring import (GAIN_KEYS, gain_fields, reject_removed_options, summarize,
                                   trial_scorer)
from models.heat_pump import get_performance
from simulation.step import SimulationInput, simulate_step

_OPTIONS = (  # name, default, must be strictly positive
    ('minimum_cop_gain', 0.0, False), ('room_tolerance_K', .01, True),
    ('storage_tolerance_K', None, True), ('min_score_gain', 1e-9, True),
    ('horizon_minutes', 720.0, True))


def _recovery_mode(options):
    mode = options.get('recovery', 'mean')
    if mode not in RECOVERY_MODES:
        raise ValueError(f'run_load_shifting.recovery must be one of {RECOVERY_MODES}')
    return mode


def validate_run_load_shifting(options):
    if 'power_step_fraction' in options:
        raise ValueError('run_load_shifting.power_step_fraction was removed; steps now follow the performance map')
    reject_removed_options('run_load_shifting', options)
    mode = _recovery_mode(options)
    for name, default, positive in _OPTIONS:
        if default is None:
            default = DEFAULT_STORAGE_TOLERANCE_K[mode]
        value = float(options.get(name, default))
        if not isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f'run_load_shifting.{name} must be finite and '
                             f'{"positive" if positive else "nonnegative"}')


def _level_cop(level, outside_C, bottom_C, params):
    hp = params.heat_pump
    sink = min(bottom_C + hp.supply_spread_K, hp.sink_temp_max_C)
    return get_performance(max(outside_C, hp.source_temp_min_C), sink, hp, level)[0]


def simulate_run_load_shifting(weather, initial, params, options):
    validate_run_load_shifting(options)
    if not params.heating_curve_control:
        raise ValueError('run_load_shifting requires a heating-curve baseline')
    mode = _recovery_mode(options)
    recovered = _recovered_mean if mode == 'mean' else _recovered
    values = {name: float(options.get(name, DEFAULT_STORAGE_TOLERANCE_K[mode] if default is None else default))
              for name, default, _ in _OPTIONS}
    interval = params.control_steps
    horizon = floor(values['horizon_minutes'] / params.step_minutes + 1e-9)
    levels = params.heat_pump.performance_map.power_fractions
    scorer = trial_scorer(options)
    results, trials = [], []
    state = initial

    def step(current, k, request=None):
        point = weather[k]
        return simulate_step(current, SimulationInput(point.outside_temperature_C,
                             point.timestamp), params, request)

    for start in range(0, len(weather), interval):
        end = min(start + interval, len(weather))
        reference = [step(state, start)]

        def reference_at(k):  # reference result of absolute step k, simulated lazily
            while start + len(reference) <= k:
                reference.append(step(reference[-1].state, start + len(reference)))
            return reference[k - start]

        def run_trial(level):
            """Simulate the raised level for one interval against the reference until recovery."""
            candidate_state, candidate = state, []
            outcome = 'timeout' if start + horizon <= len(weather) else 'no_recovery'
            record = dict(start_step=start, level=level, base_level=reference[0].power_fraction,
                          recovery_step=None, interval_results=None, **dict.fromkeys(GAIN_KEYS))
            for k in range(start, min(len(weather), start + horizon)):
                result = step(candidate_state, k, level if k < end else None)
                candidate_state = result.state
                candidate.append(result)
                if ((k < end and not isclose(result.power_fraction, level, abs_tol=1e-9))
                        or candidate_state.storage.temperature_C > params.heat_pump.sink_temp_max_C + 1e-9
                        or not all(isfinite(v) for v in (result.electric_power_W,
                            candidate_state.storage.temperature_C, candidate_state.building.T_room_C,
                            candidate_state.building.T_mass_C))
                        or result.electric_power_W < 0):
                    outcome = 'infeasible'
                    break
                if k + 1 >= end:
                    ref = reference_at(k)
                    if recovered(candidate_state, ref.state, params,
                                 values['room_tolerance_K'], values['storage_tolerance_K']):
                        reference_score = scorer.window(params, weather, start, reference[:k - start + 1])
                        record.update(recovery_step=k, interval_results=candidate[:end - start],
                                      **gain_fields(reference_score,
                                                    scorer.window(params, weather, start, candidate)))
                        outcome = 'candidate'
                        break
            record['outcome'] = outcome
            return record

        first = reference[0]
        outside = weather[start].outside_temperature_C
        base_cop = (_level_cop(first.power_fraction, outside, state.storage.bottom_C, params)
                    if first.power_fraction >= 0 else None)
        best = None
        if base_cop is not None and end - start == interval:
            for level in (l for l in levels if l > first.power_fraction + 1e-9):
                if _level_cop(level, outside, state.storage.bottom_C, params) <= base_cop + values['minimum_cop_gain']:
                    break  # higher levels do not improve the COP
                trial = run_trial(level)
                trials.append(trial)
                threshold = values['min_score_gain'] if best is None else best['score_gain']
                if trial['outcome'] != 'candidate' or trial['score_gain'] <= threshold:
                    if trial['outcome'] == 'candidate':
                        trial['outcome'] = 'no_saving'
                    break
                if best is not None:
                    best['outcome'] = 'superseded'
                best = trial
            if best is not None:
                best['outcome'] = 'accepted'

        if best is not None:
            chosen = best.pop('interval_results')
        else:
            reference_at(end - 1)
            chosen = reference[:end - start]
        for trial in trials:
            trial.pop('interval_results', None)
        results.extend(chosen)
        state = chosen[-1].state

    accepted = [t for t in trials if t['outcome'] == 'accepted']
    return results, dict(
        baseline_method=('heating_curve_supply' if params.heating_supply_curve_control else 'heating_curve'),
        saving_basis='sum_of_accepted_trial_score_gains', **summarize(accepted),
        accepted_shifts=len(accepted), interval_trials=len(trials), trials=trials,
        decision_basis='paired trial to recovery: score gain (electricity, starts, comfort)')
