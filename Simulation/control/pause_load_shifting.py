"""Heating-curve-supply baseline with bounded paired pause-extension trials."""
from math import floor, isclose, isfinite

from control.trial_scoring import (GAIN_KEYS, gain_fields, reject_removed_options, summarize,
                                   trial_scorer)
from simulation.step import SimulationInput, simulate_step


RECOVERY_MODES = ('mean', 'full')
DEFAULT_STORAGE_TOLERANCE_K = {'mean': .1, 'full': .01}


def _recovery_mode(options):
    mode = options.get('recovery', 'mean')
    if mode not in RECOVERY_MODES:
        raise ValueError(f'pause_load_shifting.recovery must be one of {RECOVERY_MODES}')
    return mode


def _level_option(options):
    level = options.get('level', 'best_cop')
    if level == 'best_cop':
        return level
    try:
        level = float(level)
    except (TypeError, ValueError):
        level = float('nan')
    if not isfinite(level) or not 0 < level <= 1:
        raise ValueError("pause_load_shifting.level must be 'best_cop' or a fraction in (0, 1]")
    return level


def validate_pause_load_shifting(options):
    reject_removed_options('pause_load_shifting', options)
    for name, default in (('room_tolerance_K', .01),
                          ('storage_tolerance_K', DEFAULT_STORAGE_TOLERANCE_K[_recovery_mode(options)]),
                          ('min_score_gain', 1e-9)):
        value = float(options.get(name, default))
        if not isfinite(value) or value <= 0:
            raise ValueError(f'pause_load_shifting.{name} must be finite and positive')
    _level_option(options)


def best_cop_level(outside_C, bottom_C, params):
    """Map level with the highest average COP at the current source/sink temperatures."""
    hp = params.heat_pump
    sink = min(bottom_C + hp.supply_spread_K, hp.sink_temp_max_C)
    curve = hp.performance_map.curves(max(outside_C, hp.source_temp_min_C), sink)
    best = max(range(len(curve)), key=lambda i: curve[i][1])
    return hp.performance_map.power_fractions[best]


def _recovered(new, old, params, room_tolerance, storage_tolerance):
    return (abs(new.building.T_room_C - old.building.T_room_C) <= room_tolerance
            and abs(new.building.T_mass_C - old.building.T_mass_C) <= room_tolerance
            and abs(params.storage.control_temperature_C(new.storage)
                    - params.storage.control_temperature_C(old.storage)) <= storage_tolerance
            and all(abs(a - b) <= storage_tolerance for a, b in zip(
                new.storage.node_temperatures_C or (new.storage.temperature_C,),
                old.storage.node_temperatures_C or (old.storage.temperature_C,))))


def _recovered_mean(new, old, params, room_tolerance, storage_tolerance):
    """Recovery on stored energy: capacity-weighted mean storage temperature plus room/mass."""
    def mean(storage):
        if storage.node_temperatures_C is None:
            return storage.temperature_C
        return params.storage.model.mean_temperature(storage.node_temperatures_C)

    return (abs(new.building.T_room_C - old.building.T_room_C) <= room_tolerance
            and abs(new.building.T_mass_C - old.building.T_mass_C) <= room_tolerance
            and abs(mean(new.storage) - mean(old.storage)) <= storage_tolerance)


def run_pause_load_shifting(weather, initial, params, options):
    validate_pause_load_shifting(options)
    if not params.heating_curve_control:
        raise ValueError('pause_load_shifting requires a heating-curve baseline')
    room_tolerance = float(options.get('room_tolerance_K', .01))
    mode = _recovery_mode(options)
    storage_tolerance = float(options.get('storage_tolerance_K', DEFAULT_STORAGE_TOLERANCE_K[mode]))
    recovered = _recovered_mean if mode == 'mean' else _recovered
    level = _level_option(options)
    minimum_gain = float(options.get('min_score_gain', 1e-9))
    scorer = trial_scorer(options)
    interval = params.control_steps
    max_steps = floor(720 / params.step_minutes + 1e-9)
    results, trials = [], []
    state = initial

    def step(current, k, request=None):
        point = weather[k]
        return simulate_step(current, SimulationInput(point.outside_temperature_C,
                             point.timestamp), params, request)

    for start in range(0, len(weather), interval):
        end = min(start + interval, len(weather))
        reference_first = step(state, start)
        reference_interval = [reference_first]
        # The actual chosen state makes a shutdown immediately after an
        # accepted extra interval eligible for another trial automatically.
        eligible = (state.heat_pump.is_on and not reference_first.state.heat_pump.is_on
                    and end - start == interval and interval <= max_steps)
        if not eligible:
            for k in range(start + 1, end):
                reference_interval.append(step(reference_interval[-1].state, k))
            results.extend(reference_interval)
            state = reference_interval[-1].state
            continue

        reference_state = state
        candidate_state = state
        candidate_interval = []
        comparison_end = min(len(weather), start + max_steps)
        extra_level = (best_cop_level(weather[start].outside_temperature_C,
                                      state.storage.bottom_C, params)
                       if level == 'best_cop' else level)
        reason = 'timeout' if start + max_steps <= len(weather) else 'no_recovery'
        recovery = None
        gains = dict.fromkeys(GAIN_KEYS)
        window_reference, window_candidate = [], []
        compared = 0
        for k in range(start, comparison_end):
            reference = reference_first if k == start else step(reference_state, k)
            reference_state = reference.state
            candidate = step(candidate_state, k, extra_level if k < end else None)
            candidate_state = candidate.state
            compared += 1
            window_reference.append(reference)
            window_candidate.append(candidate)
            if start < k < end:
                reference_interval.append(reference)
            if k < end:
                candidate_interval.append(candidate)
            if ((k < end and not isclose(candidate.power_fraction, extra_level, abs_tol=1e-9))
                    or candidate_state.storage.temperature_C > params.heat_pump.sink_temp_max_C + 1e-9
                    or not all(isfinite(v) for v in (candidate.electric_power_W,
                        candidate_state.storage.temperature_C, candidate_state.building.T_room_C))
                    or not isfinite(candidate_state.building.T_mass_C)
                    or candidate.electric_power_W < 0):
                reason = 'infeasible'
                break
            if k + 1 >= end and recovered(candidate_state, reference_state, params,
                                          room_tolerance, storage_tolerance):
                recovery = k
                gains = gain_fields(scorer.window(params, weather, start, window_reference),
                                    scorer.window(params, weather, start, window_candidate))
                reason = 'accepted' if gains['score_gain'] > minimum_gain else 'no_saving'
                break

        # If equipment constraints stopped the trial early, finish only the
        # ordinary interval. All later comparison states are discarded.
        for k in range(start + len(reference_interval), end):
            reference_interval.append(step(reference_interval[-1].state, k))
        chosen = candidate_interval if reason == 'accepted' else reference_interval
        results.extend(chosen)
        state = chosen[-1].state
        trials.append(dict(start_step=start, recovery_step=recovery, level=extra_level,
                           outcome=reason, comparison_minutes=compared * params.step_minutes, **gains))

    accepted = [t for t in trials if t['outcome'] == 'accepted']
    return results, dict(baseline_method=('heating_curve_supply' if params.heating_supply_curve_control
                                            else 'heating_curve'),
                        saving_basis='sum_of_accepted_trial_score_gains',
                        **summarize(accepted),
                        accepted_shifts=len(accepted), interval_trials=len(trials), trials=trials)
