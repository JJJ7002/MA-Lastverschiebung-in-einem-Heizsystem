from datetime import datetime, timedelta
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support import load_test_config
from weather.loader import WeatherPoint
from control.run_load_shifting import validate_run_load_shifting
from simulation.engine import run_simulation


def scenario():
    config = load_test_config()
    config['weather']['source'] = 'csv'  # Isolate model tests from network access
    config['simulation']['weather_file'] = 'data/weather/ogd-smn_bas_t_historical_2020-2029.csv'
    config['simulation'].update(start_time=None, end_time=None, settling_hours=0)
    config['control'].update(method='run_load_shifting', interval_minutes=10)
    for name in ('pause_load_shifting', 'run_load_shifting'):  # independent of the yaml defaults
        config['control'].setdefault(name, {})['base_method'] = 'heating_curve_supply'
    config['control']['night_setback']['enabled'] = False
    config['heat_pump'].update(min_on_time_min=30, min_off_time_min=30)
    config['storage'].update(T_storage_init_C=40, UA_storage_W_per_K=2)
    config['scoring'].update(cost_per_heat_pump_start=0., cost_per_comfort_degree_hour=0.)  # electricity only
    weather = [WeatherPoint(datetime(2020, 1, 1) + timedelta(minutes=2 * k),
                            15 if k < 35 else -7) for k in range(720)]
    return config, weather


@pytest.mark.parametrize('name', ['room_tolerance_K', 'storage_tolerance_K', 'min_score_gain',
                                  'horizon_minutes'])
@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_positive_options_are_validated(name, value):
    with pytest.raises(ValueError, match=name):
        validate_run_load_shifting({name: value})


@pytest.mark.parametrize('name', ['minimum_cop_gain'])
@pytest.mark.parametrize('value', [-1, float('nan'), float('inf')])
def test_nonnegative_options_are_validated(name, value):
    with pytest.raises(ValueError, match=name):
        validate_run_load_shifting({name: value})


def test_invalid_recovery_mode_is_rejected():
    with pytest.raises(ValueError, match='recovery'):
        validate_run_load_shifting({'recovery': 'strict'})


def test_without_candidates_the_baseline_is_reproduced_exactly():
    config, weather = scenario()
    config['control']['run_load_shifting']['minimum_cop_gain'] = 100.
    shifted = run_simulation(config, weather_points=weather)
    config['control']['method'] = 'heating_curve_supply'
    baseline = run_simulation(config, weather_points=weather)
    assert shifted.load_shifting_summary['interval_trials'] == 0
    assert shifted.total_electric_energy_Wh == pytest.approx(baseline.total_electric_energy_Wh)
    assert shifted.room_temperatures_C == baseline.room_temperatures_C


@pytest.mark.parametrize('name', ['min_saving_Wh', 'start_penalty_Wh', 'comfort_penalty_Wh_per_Kh',
                                  'credit_residual_energy'])
def test_options_of_the_removed_wh_criterion_are_rejected(name):
    with pytest.raises(ValueError, match=f'{name} was removed'):
        validate_run_load_shifting({name: 1.})


def tolerant(config):
    config['control']['run_load_shifting'].update(room_tolerance_K=.05, storage_tolerance_K=.2)
    return config


def test_trial_bookkeeping_is_consistent():
    config, weather = scenario()
    config['scoring']['electricity']['default_price_per_kWh'] = 1e-3  # score gain == saved Wh / 1e6
    result = run_simulation(tolerant(config), weather_points=weather)
    summary = result.load_shifting_summary
    assert len(result.room_temperatures_C) == len(weather)
    assert summary['interval_trials'] == len(summary['trials']) > 0
    assert summary['baseline_method'] == 'heating_curve_supply'
    accepted = [t for t in summary['trials'] if t['outcome'] == 'accepted']
    assert summary['accepted_shifts'] == len(accepted)
    # At most one accepted level per interval, and it beats the baseline level and the threshold.
    assert len({t['start_step'] for t in accepted}) == len(accepted)
    for trial in summary['trials']:
        if trial['score_gain'] is not None:
            assert trial['score_gain'] == pytest.approx(
                trial['electricity_gain'] + trial['start_gain'] + trial['comfort_gain'])
            assert trial['electricity_gain'] == pytest.approx(trial['saved_electric_energy_Wh'] * 1e-6)
    for trial in accepted:
        assert trial['level'] > trial['base_level']
        assert trial['score_gain'] > 1e-9
        assert result.heat_pump_power_fractions[trial['start_step']] == pytest.approx(trial['level'])
    assert summary['score_gain'] == pytest.approx(sum(t['score_gain'] for t in accepted))


def test_valued_starts_only_accept_trials_that_do_not_add_starts():
    config, weather = scenario()
    config['scoring']['cost_per_heat_pump_start'] = 1e3
    summary = run_simulation(tolerant(config), weather_points=weather).load_shifting_summary
    assert all(t['extra_starts'] <= 0 for t in summary['trials'] if t['outcome'] == 'accepted')
    assert summary['start_gain'] >= 0


def _base_method_scenario():
    config, weather = scenario()
    # A warm spell lets the baseline heat pump shut down, which also creates pause trials.
    weather = [WeatherPoint(p.timestamp, 20 if 280 <= k < 300 else p.outside_temperature_C)
               for k, p in enumerate(weather)]
    return config, weather


@pytest.mark.parametrize('method', ['pause_load_shifting', 'run_load_shifting'])
def test_base_method_selects_the_underlying_control(method):
    for base in ('heating_curve_supply', 'heating_curve'):
        config, weather = _base_method_scenario()
        config['control']['method'] = method
        config['control'].setdefault(method, {})['base_method'] = base
        result = run_simulation(config, weather_points=weather)
        assert result.load_shifting_summary['baseline_method'] == base

    config, weather = _base_method_scenario()
    config['control']['method'] = method
    config['control'].setdefault(method, {})['base_method'] = 'cop_optimized'
    with pytest.raises(ValueError, match='base_method'):
        run_simulation(config, weather_points=weather)
