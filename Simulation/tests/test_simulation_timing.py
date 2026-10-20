from datetime import datetime, timedelta
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support import load_test_config
from weather.loader import WeatherPoint, resample_weather
from simulation.engine import build_simulation_params, initial_state, run_simulation
from simulation.step import SimulationInput, simulate_step
from control.heating_curve import heating_curve_power_request
from analysis.metrics import calculate_metrics
from analysis.scoring import ScoringParams, calculate_score


def setup():
    config = load_test_config()
    config['weather']['source'] = 'csv'  # Isolate model tests from network access
    config['simulation']['weather_file'] = 'data/weather/ogd-smn_bas_t_historical_2020-2029.csv'
    config['simulation'].update(dt_minutes=2, start_time=None, end_time=None)
    config['control'].update(method='heating_curve', interval_minutes=10)
    config['control']['schedule']['enabled'] = False
    config['control']['night_setback']['enabled'] = False
    config['heat_pump'].update(min_on_time_min=0, min_off_time_min=0)
    weather = [WeatherPoint(datetime(2020, 1, 1) + timedelta(minutes=10*i), -5+i)
               for i in range(3)]
    return config, weather


def test_weather_duration_and_energy_are_preserved():
    config, weather = setup()
    with patch('simulation.step.heating_curve_power_request', wraps=heating_curve_power_request) as control:
        result = run_simulation(config, weather_points=weather)
    assert control.call_count == 3
    assert len(result.timestamps) == 15
    assert result.timestamps[-1] == '2020-01-01 00:28:00'
    assert result.outside_temperatures_C == [-5]*5 + [-4]*5 + [-3]*5
    assert result.total_electric_energy_Wh == pytest.approx(sum(result.electric_power_W) * 2 / 60)
    assert result.total_thermal_energy_Wh == pytest.approx(sum(result.thermal_power_W) * 2 / 60)
    for i in range(0, 15, 5):
        assert len(set(result.heat_pump_power_fractions[i:i+5])) == 1


def test_hold_advances_physics_and_timer_then_accepts_next_request():
    config, weather = setup()
    params = build_simulation_params(config)
    state = initial_state(config, -5)
    first = simulate_step(state, SimulationInput(-5, weather[0].timestamp), params, .6)
    state = first.state
    for i in range(1, 5):
        state = simulate_step(state, SimulationInput(-5, weather[0].timestamp + timedelta(minutes=2*i)),
                              params, -1).state
        assert state.power_fraction == .6
    assert state.heat_pump.minutes_in_current_state == pytest.approx(8)
    assert state.storage.temperature_C != first.state.storage.temperature_C
    state = simulate_step(state, SimulationInput(-4, weather[1].timestamp), params, -1).state
    assert state.power_fraction == -1


@pytest.mark.parametrize('interval', [0, -2, 1, 3, float('nan'), float('inf')])
def test_invalid_control_interval(interval):
    config, _ = setup()
    config['control']['interval_minutes'] = interval
    with pytest.raises(ValueError, match='interval_minutes'):
        build_simulation_params(config)


def test_legacy_config_and_weather_validation():
    config, weather = setup()
    del config['control']['interval_minutes']
    assert build_simulation_params(config).control_steps == 1
    with pytest.raises(ValueError, match='Weather interval'):
        resample_weather(weather, 3 / 60)
    with pytest.raises(ValueError, match='Weather interval'):
        resample_weather(list(reversed(weather)), 2 / 60)


def test_time_window_can_start_inside_weather_interval():
    config, weather = setup()
    config['simulation'].update(start_time='2020-01-01 00:04', end_time='2020-01-01 00:14')
    result = run_simulation(config, weather_points=weather)
    assert len(result.timestamps) == 5
    assert result.timestamps[0] == '2020-01-01 00:04:00'
    assert result.timestamps[-1] == '2020-01-01 00:12:00'


def test_rl_control_interval_reward_matches_inference():
    pytest.importorskip('gymnasium')
    from learning.environment import HeatPumpEnv
    config, weather = setup()
    env = HeatPumpEnv(config, weather, episode_steps=10, random_start=False)
    env.reset()
    reward_sum = 0
    for index, action in enumerate([4, 0]):
        _, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward
        assert info['currency'] == config['scoring']['currency']
        assert not terminated
        assert truncated == (index == 1)
    config['control']['method'] = 'reinforcement_learning'
    with patch('control.rl_power.RLController') as controller:
        controller.return_value.predict.side_effect = [env.power_levels[4], env.power_levels[0]]
        result = run_simulation(config, weather_points=weather[:2])
        assert controller.return_value.predict.call_count == 2
    metrics = calculate_metrics(result, config['building']['comfort_buffer_K'], 2 / 60,
                                config['building']['T_set_C'])
    score = calculate_score(result, metrics, 2 / 60, ScoringParams.from_config(config['scoring']))
    assert -reward_sum == pytest.approx(score.total_score)
    assert env.state.storage.temperature_C == pytest.approx(result.storage_temperatures_C[-1])
    assert env.state.building.T_room_C == pytest.approx(result.room_temperatures_C[-1])


@pytest.mark.parametrize('interval, calls', [(2, 15), (6, 5), (10, 3)])
def test_configurable_control_cadence(interval, calls):
    config, weather = setup()
    config['control']['interval_minutes'] = interval
    with patch('simulation.step.heating_curve_power_request', wraps=heating_curve_power_request) as controller:
        result = run_simulation(config, weather_points=weather)
    assert len(result.timestamps) == 15
    assert controller.call_count == calls



def test_cop_controller_uses_control_cadence():
    from control.cop_optimized import update_cop_control
    config, weather = setup()
    config['control']['method'] = 'cop_optimized'
    with patch('simulation.engine.load_daily_temperatures', return_value={weather[0].timestamp.date(): -4}), \
         patch('simulation.step.update_cop_control', wraps=update_cop_control) as controller:
        result = run_simulation(config, weather_points=weather)
    assert controller.call_count == 3
    assert len(result.timestamps) == 15
    assert all(value is not None for value in result.reference_cop)
