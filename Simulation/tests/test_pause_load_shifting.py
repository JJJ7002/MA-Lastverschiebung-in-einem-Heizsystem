from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support import load_test_config
from weather.loader import WeatherPoint
from simulation.engine import build_simulation_params, initial_state, run_simulation

# The deterministic stand-in plant always applies 0.75 and the tests patch the full check.
LEGACY = {'recovery': 'full', 'level': .75}
from simulation.step import SimulationInput, simulate_step
from control.pause_load_shifting import run_pause_load_shifting


def scenario():
    config = load_test_config()
    config['weather']['source'] = 'csv'  # Isolate model tests from network access
    config['simulation']['weather_file'] = 'data/weather/ogd-smn_bas_t_historical_2020-2029.csv'
    config['simulation'].update(start_time=None, end_time=None, settling_hours=0)
    config['control'].update(method='heating_curve_supply', interval_minutes=10)
    config['control']['night_setback']['enabled'] = False
    config['heat_pump'].update(min_on_time_min=30, min_off_time_min=30)
    config['control']['pause_load_shifting']['base_method'] = 'heating_curve_supply'  # independent of the yaml default
    # Electricity-only scoring here; starts and comfort have their own tests below.
    config['control']['pause_load_shifting'].update(room_tolerance_K=.01, storage_tolerance_K=.01)
    config['scoring'].update(cost_per_heat_pump_start=0., cost_per_comfort_degree_hour=0.)  # electricity only
    config['storage'].update(T_storage_init_C=40, UA_storage_W_per_K=2)
    weather = [WeatherPoint(datetime(2020, 1, 1) + timedelta(minutes=2*k),
                            15 if k < 35 else -7) for k in range(720)]
    return config, weather


def test_engine_uses_changed_reference_and_preserves_physical_replay():
    config, weather = scenario()
    weather = [WeatherPoint(p.timestamp, 20 if 280 <= k < 300 else p.outside_temperature_C)
               for k, p in enumerate(weather)]
    baseline = run_simulation(config, weather_points=weather)
    config['control']['method'] = 'pause_load_shifting'
    result = run_simulation(config, weather_points=weather)
    summary = result.load_shifting_summary
    assert summary['interval_trials'] > 0
    accepted = [trial for trial in summary['trials'] if trial['outcome'] == 'accepted']
    assert len(accepted) == summary['accepted_shifts']
    for trial in accepted:
        assert result.heat_pump_power_fractions[trial['start_step']] == pytest.approx(trial['level'])
    assert all(trial['recovery_step'] > trial['start_step'] + 5 for trial in accepted)
    assert result.total_electric_energy_Wh <= baseline.total_electric_energy_Wh
    state = initial_state(config, 15)
    params = build_simulation_params(config)
    for k, point in enumerate(weather):
        step = simulate_step(state, SimulationInput(point.outside_temperature_C, point.timestamp),
                             params, result.hydraulics[k]['heat_pump_commanded_power_fraction'])
        state = step.state
        assert state.storage.temperature_C == pytest.approx(result.storage_temperatures_C[k])
        assert state.building.T_room_C == pytest.approx(result.room_temperatures_C[k])
    assert result.total_electric_energy_Wh == pytest.approx(sum(result.electric_power_W) * 2 / 60)
    assert result.total_thermal_energy_Wh == pytest.approx(sum(result.thermal_power_W) * 2 / 60)


def baseline_steps():
    config, weather = scenario()
    params = build_simulation_params(config)
    state = initial = initial_state(config, 15)
    results = []
    for point in weather:
        result = simulate_step(state, SimulationInput(point.outside_temperature_C, point.timestamp), params)
        results.append(result)
        state = result.state
    return params, initial, weather, results


def test_rejected_trials_stop_at_twelve_hours_without_simulating_suffix():
    params, initial, weather, results = baseline_steps()
    weather = weather + [WeatherPoint(p.timestamp + timedelta(days=1), p.outside_temperature_C) for p in weather]
    results = []
    state = initial
    for point in weather:
        result = simulate_step(state, SimulationInput(point.outside_temperature_C, point.timestamp), params)
        results.append(result)
        state = result.state
    with patch('control.pause_load_shifting._recovered', return_value=False), patch(
            'control.pause_load_shifting.simulate_step', wraps=simulate_step) as physics:
        optimized, summary = run_pause_load_shifting(weather, initial, params, LEGACY)
    assert optimized == results
    assert summary['trials'][0]['outcome'] in ('timeout', 'infeasible')
    assert all(t['comparison_minutes'] <= 720 for t in summary['trials'])
    assert physics.call_count <= len(weather) + sum(
        2*t['comparison_minutes']/params.step_minutes for t in summary['trials'])


def test_rejected_trial_stops_immediately_on_temperature_match():
    params, initial, weather, results = baseline_steps()
    with patch('control.pause_load_shifting._recovered', return_value=True), patch(
            'control.pause_load_shifting.simulate_step', wraps=simulate_step) as physics:
        optimized, summary = run_pause_load_shifting(weather, initial, params, LEGACY)
    assert optimized == results
    assert all(t['outcome'] == 'no_saving' for t in summary['trials'])
    assert physics.call_count == len(weather) + summary['interval_trials'] * params.control_steps


def test_rejection_does_not_skip_shutdowns_during_comparison():
    config, weather = scenario()
    params = build_simulation_params(config)
    initial = state = initial_state(config, 15)
    expected = []
    for k, point in enumerate(weather):
        result = simulate_step(state, SimulationInput(point.outside_temperature_C, point.timestamp), params)
        if (k % params.control_steps == 0 and state.heat_pump.is_on
                and not result.state.heat_pump.is_on):
            expected.append(k)
        state = result.state
    with patch('control.pause_load_shifting._recovered', return_value=False):
        _, summary = run_pause_load_shifting(weather, initial, params, LEGACY)
    actual = [t['start_step'] for t in summary['trials']]
    assert actual == expected
    if len(actual) > 1:
        assert actual[1] < actual[0] + 360


def test_rejects_trials_without_recovery_and_keeps_input_unchanged():
    config, weather = scenario()
    params = build_simulation_params(config)
    state = initial = initial_state(config, 15)
    results = []
    for point in weather:
        result = simulate_step(state, SimulationInput(point.outside_temperature_C, point.timestamp), params)
        results.append(result)
        state = result.state
    original = deepcopy(results)
    optimized, summary = run_pause_load_shifting(weather, initial, params,
        dict(room_tolerance_K=1e-14, storage_tolerance_K=1e-14, min_score_gain=1e12))
    assert summary['accepted_shifts'] == 0
    assert optimized == results == original


def test_unprofitable_extension_is_rejected():
    config, weather = scenario()
    baseline = run_simulation(config, weather_points=weather)
    config['control']['method'] = 'pause_load_shifting'
    config['control']['pause_load_shifting']['min_score_gain'] = 1e12
    result = run_simulation(config, weather_points=weather)
    assert result.load_shifting_summary['accepted_shifts'] == 0
    assert all(t['outcome'] in ('no_saving', 'infeasible')
               for t in result.load_shifting_summary['trials'])
    assert result.electric_power_W == baseline.electric_power_W


def test_short_horizon_cannot_accept_unrecovered_heat():
    config, weather = scenario()
    weather = [WeatherPoint(p.timestamp, 20 if 280 <= k < 300 else p.outside_temperature_C)
               for k, p in enumerate(weather[:325])]
    baseline = run_simulation(config, weather_points=weather)
    config['control']['method'] = 'pause_load_shifting'
    with patch('control.pause_load_shifting._recovered', return_value=False):
        result = run_simulation(config, weather_points=weather)
    assert result.load_shifting_summary['trials'][0]['outcome'] in ('no_recovery', 'infeasible')
    assert result.electric_power_W == baseline.electric_power_W


@pytest.mark.parametrize('name,value', [('recovery', 'strict'), ('level', 'max'), ('level', 0),
                                        ('level', 1.5), ('level', float('nan'))])
def test_invalid_recovery_and_level_options(name, value):
    config, weather = scenario()
    config['control']['method'] = 'pause_load_shifting'
    config['control']['pause_load_shifting'][name] = value
    with pytest.raises(ValueError, match=name):
        run_simulation(config, weather_points=weather)


@pytest.mark.parametrize('name', ['room_tolerance_K', 'storage_tolerance_K', 'min_score_gain'])
@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_invalid_options(name, value):
    config, weather = scenario()
    config['control']['method'] = 'pause_load_shifting'
    config['control']['pause_load_shifting'][name] = value
    with pytest.raises(ValueError, match=name):
        run_simulation(config, weather_points=weather)


def algorithm_fixture(cyclic):
    """Deterministic stand-in to exercise optimizer decisions independently of a plant."""
    from dataclasses import replace
    from models.storage import StorageState
    from control.operating_constraints import OperatingState
    from simulation.step import StepResult
    config, _ = scenario()
    params = build_simulation_params(config)
    initial = initial_state(config, 15)
    weather = [WeatherPoint(datetime(2020, 1, 1) + timedelta(minutes=2*k), 0)
               for k in range(400)]

    def physics(state, inputs, params, request=None):
        k = state.steps_elapsed
        stored = state.storage.temperature_C - 40
        normal_on = k % 10 < 5 if cyclic else k < 5
        extra_on = request is not None and request > 0
        if extra_on:
            stored += 1
            electric, fraction = .5, .75
        elif normal_on and stored > 0:
            stored -= 1
            electric, fraction = 0, 1
        else:
            electric, fraction = (10, 1) if normal_on else (0, 0)
        new = replace(state, storage=StorageState(40 + stored, (40 + stored,) * 10),
                      heat_pump=OperatingState(normal_on or extra_on, 0),
                      steps_elapsed=k + 1,
                      total_electric_energy_Wh=state.total_electric_energy_Wh + electric * params.dt_hours)
        return StepResult(new, 1, electric, electric, 0, fraction, 0)
    return params, initial, weather, physics


def test_paired_trial_can_accept_a_known_saving():
    params, initial, weather, physics = algorithm_fixture(True)
    with patch('control.pause_load_shifting.simulate_step', side_effect=physics):
        _, summary = run_pause_load_shifting(weather[:40], initial, params, LEGACY)
    assert summary['accepted_shifts'] > 0
    assert summary['saved_electric_energy_Wh'] > 0
    assert all(t['recovery_step'] >= t['start_step'] + params.control_steps
               for t in summary['trials'] if t['outcome'] == 'accepted')


def test_feasible_trial_without_recovery_stops_after_twelve_hours():
    params, initial, weather, physics = algorithm_fixture(False)
    with patch('control.pause_load_shifting.simulate_step', side_effect=physics), patch(
            'control.pause_load_shifting._recovered', return_value=False):
        _, summary = run_pause_load_shifting(weather, initial, params, LEGACY)
    assert summary['trials'][0]['outcome'] == 'timeout'
    assert summary['trials'][0]['comparison_minutes'] == 720


def test_equal_mean_but_different_stratification_is_not_recovered():
    from dataclasses import replace
    from models.storage import StorageState
    from control.pause_load_shifting import _recovered
    params, initial, _, _ = algorithm_fixture(False)
    stratified = replace(initial, storage=StorageState(40, (45,) * 5 + (35,) * 5))
    assert stratified.storage.mean_temperature_C == initial.storage.mean_temperature_C
    assert not _recovered(stratified, initial, params, .01, .01)


def test_rejected_comparisons_keep_every_later_shutdown_eligible():
    params, initial, weather, physics = algorithm_fixture(True)
    with patch('control.pause_load_shifting.simulate_step', side_effect=physics), patch(
            'control.pause_load_shifting._recovered', return_value=False):
        _, summary = run_pause_load_shifting(weather[:60], initial, params, LEGACY)
    assert [t['start_step'] for t in summary['trials']] == list(range(5, 60, 10))


def test_mean_recovery_ignores_stratification_but_not_energy():
    from dataclasses import replace
    from models.storage import StorageState
    from control.pause_load_shifting import _recovered_mean
    params, initial, _, _ = algorithm_fixture(False)
    stratified = replace(initial, storage=StorageState(40, (45,) * 5 + (35,) * 5))
    assert _recovered_mean(stratified, initial, params, .01, .01)
    warmer = replace(initial, storage=StorageState(41, (41,) * 10))
    assert not _recovered_mean(warmer, initial, params, .01, .01)
    hotter_room = replace(initial, building=replace(initial.building,
                          T_room_C=initial.building.T_room_C + .5))
    assert not _recovered_mean(hotter_room, initial, params, .01, 10)


def test_best_cop_level_is_a_map_level_with_maximal_cop():
    from control.pause_load_shifting import best_cop_level
    params, initial, _, _ = algorithm_fixture(False)
    hp = params.heat_pump
    level = best_cop_level(5., 30., params)
    curve = hp.performance_map.curves(5., min(30. + hp.supply_spread_K, hp.sink_temp_max_C))
    assert level in hp.performance_map.power_fractions
    assert curve[list(hp.performance_map.power_fractions).index(level)][1] == max(c[1] for c in curve)


@pytest.mark.parametrize('name', ['min_saving_Wh', 'start_penalty_Wh', 'comfort_penalty_Wh_per_Kh',
                                  'comfort_buffer_K', 'credit_residual_energy'])
def test_options_of_the_removed_wh_criterion_are_rejected(name):
    config, weather = scenario()
    config['control']['method'] = 'pause_load_shifting'
    config['control']['pause_load_shifting'][name] = 1.
    with pytest.raises(ValueError, match=f'{name} was removed'):
        run_simulation(config, weather_points=weather)


def room_follows_storage_fixture():
    """Stand-in plant whose room temperature rises with the stored heat (below the comfort threshold)."""
    from dataclasses import replace
    params, initial, weather, physics = algorithm_fixture(True)

    def room_physics(state, inputs, params, request=None):
        result = physics(state, inputs, params, request)
        building = replace(result.state.building,
                           T_room_C=19.5 + .1 * (result.state.storage.temperature_C - 40))
        return replace(result, state=replace(result.state, building=building))
    return params, initial, weather, room_physics


def fixture_scorer(comfort_cost, price=1.0, start_cost=0.0):
    from analysis.scoring import ScoringParams
    from control.trial_scoring import TrialScorer
    return TrialScorer(ScoringParams.from_config(dict(
        currency='CHF', cost_per_heat_pump_start=start_cost, cost_per_comfort_degree_hour=comfort_cost,
        electricity=dict(default_price_per_kWh=price, tariff_windows_enabled=False))), comfort_buffer_K=.1)


def run_room_fixture(comfort_cost, **options):
    params, initial, weather, physics = room_follows_storage_fixture()
    with patch('control.pause_load_shifting.simulate_step', side_effect=physics):
        return run_pause_load_shifting(weather[:40], initial, params, dict(
            LEGACY, scorer=fixture_scorer(comfort_cost), **options))[1]


def test_trial_score_gain_is_electricity_plus_start_plus_comfort():
    for comfort_cost in (0., 5., 1000.):
        summary = run_room_fixture(comfort_cost)
        measured = [t for t in summary['trials'] if t['score_gain'] is not None]
        assert measured
        for trial in measured:
            assert trial['comfort_deficit_change_Kh'] == pytest.approx(-.0833333, abs=1e-4)
            assert trial['comfort_gain'] == pytest.approx(-comfort_cost * trial['comfort_deficit_change_Kh'])
            assert trial['electricity_gain'] == pytest.approx(trial['saved_electric_energy_Wh'] / 1000)
            assert trial['score_gain'] == pytest.approx(
                trial['electricity_gain'] + trial['start_gain'] + trial['comfort_gain'])
        accepted = [t for t in summary['trials'] if t['outcome'] == 'accepted']
        assert summary['score_gain'] == pytest.approx(sum(t['score_gain'] for t in accepted))
        assert summary['comfort_gain'] == pytest.approx(sum(t['comfort_gain'] for t in accepted))


def test_avoided_discomfort_can_turn_a_rejected_trial_into_an_accepted_one():
    assert run_room_fixture(0., min_score_gain=1.)['accepted_shifts'] == 0
    assert run_room_fixture(100., min_score_gain=1.)['accepted_shifts'] > 0


def test_added_discomfort_rejects_an_electrically_profitable_trial():
    from dataclasses import replace
    params, initial, weather, physics = room_follows_storage_fixture()

    def hotter_is_worse(state, inputs, params, request=None):  # the extra heat lowers the room
        result = physics(state, inputs, params, request)
        building = replace(result.state.building, T_room_C=22.5 - .1 * (result.state.storage.temperature_C - 40)
                           - (1.5 if request else 0))
        return replace(result, state=replace(result.state, building=building))
    options = dict(LEGACY, room_tolerance_K=10.)
    with patch('control.pause_load_shifting.simulate_step', side_effect=hotter_is_worse):
        free = run_pause_load_shifting(weather[:40], initial, params,
                                       dict(options, scorer=fixture_scorer(0.)))[1]
        costly = run_pause_load_shifting(weather[:40], initial, params,
                                         dict(options, scorer=fixture_scorer(1e6)))[1]
    assert free['accepted_shifts'] > 0
    assert costly['accepted_shifts'] < free['accepted_shifts']
    assert all(t['comfort_deficit_change_Kh'] <= 1e-9 for t in costly['trials'] if t['outcome'] == 'accepted')
