from dataclasses import replace
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip('gymnasium')
pytest.importorskip('stable_baselines3')
from stable_baselines3.common.env_checker import check_env
from support import load_test_config
from analysis.metrics import calculate_metrics
from analysis.scoring import calculate_score, ScoringParams
from control.heating_curve import heating_curve_target
from control.rl_offset import (OFFSET_OBSERVATION_VERSION, OffsetSpec, RLOffsetController,
                               offset_observation)
from weather.loader import WeatherPoint
from learning.environment import HeatPumpEnv
from simulation.engine import build_simulation_params, initial_state, run_simulation
from simulation.step import SimulationInput, simulate_step
from learning.training import load_training_settings, train


def setup():
    config = load_test_config()
    config['weather']['source'] = 'csv'
    config['simulation']['weather_file'] = 'data/weather/ogd-smn_bas_t_historical_2020-2029.csv'
    config['simulation']['dt_minutes'] = 10
    config['simulation'].update(start_time=None, end_time=None, settling_hours=0)
    config['control']['method'] = 'reinforcement_learning_offset'
    weather = [WeatherPoint(datetime(2020, 1, 1, 20) + timedelta(minutes=10 * i), -5)
               for i in range(145)]
    return config, weather


def test_shift_gain_follows_measured_wpm_values_and_is_interpolated():
    config, _ = setup()
    params = build_simulation_params(config)
    # Measured: comfort 22 vs eco 19 degC (3 K) moves the supply target by 4.6 / 5.0 / 5.8 K
    for outside, measured in ((-20, 4.6), (0, 5.0), (20, 5.8)):
        assert 3 * params.hk_shift_gain(outside) == pytest.approx(measured, abs=0.01)
    assert params.hk_shift_gain(-10) == pytest.approx((1.533 + 1.667) / 2)
    assert params.hk_shift_gain(-40) == pytest.approx(1.533)  # constant outside the measured range
    assert params.hk_shift_gain(35) == pytest.approx(1.933)


def test_shift_gain_validation():
    config, _ = setup()
    config['control']['heating_curve']['hk_shift_gain'] = {'outside_C': [0, 10], 'K_per_K': [1.5]}
    with pytest.raises(ValueError, match='hk_shift_gain'):
        build_simulation_params(config)
    config['control']['heating_curve']['hk_shift_gain'] = {'outside_C': [10, 0], 'K_per_K': [1.5, 1.6]}
    with pytest.raises(ValueError, match='hk_shift_gain'):
        build_simulation_params(config)


def test_target_moves_by_shift_times_gain_and_zero_shift_changes_nothing():
    config, _ = setup()
    params = build_simulation_params(config)
    timestamp = datetime(2020, 1, 1, 12)
    for outside in (-5, 0, 12):
        base = heating_curve_target(outside, timestamp, params)
        assert heating_curve_target(outside, timestamp, params, 0.0) == base
        assert (heating_curve_target(outside, timestamp, params, 2.0)
                == pytest.approx(base + 2 * params.hk_shift_gain(outside)))
        assert (heating_curve_target(outside, timestamp, params, -3.0)
                == pytest.approx(base - 3 * params.hk_shift_gain(outside)))
    cap = params.heat_pump.sink_temp_max_C
    assert heating_curve_target(-20, timestamp, params, 40.0) == cap


def _run_with_constant_shift(config, weather, shift):
    with patch('control.rl_offset.RLOffsetController') as controller:
        controller.return_value.shift_K.return_value = shift
        return run_simulation(config, weather_points=weather)


def test_zero_shift_is_bit_identical_to_heating_curve_and_shift_changes_the_target():
    config, weather = setup()
    offset = _run_with_constant_shift(config, weather, 0.0)
    config['control']['method'] = 'heating_curve'
    plain = run_simulation(config, weather_points=weather)
    assert offset.room_temperatures_C == plain.room_temperatures_C
    assert offset.electric_power_W == plain.electric_power_W
    assert [r['buffer_target_C'] for r in offset.hydraulics] == [r['buffer_target_C'] for r in plain.hydraulics]
    config['control']['method'] = 'reinforcement_learning_offset'
    warmer = _run_with_constant_shift(config, weather, 3.0)
    gain = build_simulation_params(config).hk_shift_gain(-5)
    differences = [w['buffer_target_C'] - p['buffer_target_C']
                   for w, p in zip(warmer.hydraulics, plain.hydraulics)]
    assert differences[0] == pytest.approx(3 * gain)
    assert all(row['hk_shift_K'] == 3.0 for row in warmer.hydraulics)
    assert sum(warmer.electric_power_W) > sum(plain.electric_power_W)


def test_constant_shift_reference_needs_no_model_and_zero_equals_heating_curve():
    config, weather = setup()
    section = config['control']['reinforcement_learning_offset']
    section['model_path'] = 'results/does_not_exist/model.zip'
    section['constant_shift_K'] = 0
    zero = run_simulation(config, weather_points=weather)
    config['control']['method'] = 'heating_curve'
    plain = run_simulation(config, weather_points=weather)
    assert zero.room_temperatures_C == plain.room_temperatures_C
    assert zero.electric_power_W == plain.electric_power_W
    config['control']['method'] = 'reinforcement_learning_offset'
    section['constant_shift_K'] = 2
    warmer = run_simulation(config, weather_points=weather)
    assert {row['hk_shift_K'] for row in warmer.hydraulics} == {2.0}
    assert sum(warmer.electric_power_W) > sum(plain.electric_power_W)
    section['constant_shift_K'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        run_simulation(config, weather_points=weather)


def test_offset_spec_actions_and_observation_options():
    config, _ = setup()
    spec = OffsetSpec.from_config(config)
    assert spec.shifts_K[0] == -4.0 and spec.shifts_K[-1] == 4.0 and len(spec.shifts_K) == 9
    assert spec.observation_size == 9
    section = config['control']['reinforcement_learning_offset']
    section['observation'] = {'last_shift': False, 'heat_pump_power': False}
    section['shift_K'] = [-2, 0, 2]
    spec = OffsetSpec.from_config(config)
    assert spec.shifts_K == (-2.0, 0.0, 2.0) and spec.observation_size == 7
    section['shift_K'] = [0, 0]
    with pytest.raises(ValueError, match='increasing'):
        OffsetSpec.from_config(config)
    section['shift_K'] = {'max': 1, 'step': 0}
    with pytest.raises(ValueError, match='step'):
        OffsetSpec.from_config(config)
    section['shift_K'] = {'max': 2.5, 'step': 1}
    with pytest.raises(ValueError, match='multiple'):
        OffsetSpec.from_config(config)
    section['shift_K'] = {'max': 1.0, 'step': 0.5}
    assert OffsetSpec.from_config(config).shifts_K == (-1.0, -0.5, 0.0, 0.5, 1.0)
    section['shift_K'] = [-1, 1]
    section['observation'] = {'unknown': True}
    with pytest.raises(ValueError, match='Unknown'):
        OffsetSpec.from_config(config)


def test_observation_contains_only_measurable_values_and_the_controllers_own_state():
    config, weather = setup()
    params = build_simulation_params(config)
    spec = OffsetSpec.from_config(config)
    state = initial_state(config, -5)
    point = weather[0]
    values = offset_observation(state, point.outside_temperature_C, point.timestamp, params, spec)
    assert len(values) == spec.observation_size == 9
    assert values[0] == pytest.approx(-5 / 20)
    target = params.room_target_C_at(point.timestamp)
    assert values[1] == pytest.approx((state.building.T_room_C - target) / 0.5)
    assert values[2] == pytest.approx(target - 20)
    assert values[5] == pytest.approx((params.storage.control_temperature_C(state.storage) - 40) / 20)
    assert values[7] == pytest.approx(0.0)
    assert values[8] == state.power_fraction
    shifted = offset_observation(replace(state, hk_shift_K=4.0), point.outside_temperature_C,
                                 point.timestamp, params, spec)
    assert shifted[7] == pytest.approx(1.0)


def test_env_offset_mode_reward_matches_engine_run():
    config, weather = setup()
    env = HeatPumpEnv(config, weather, random_start=False)
    check_env(env)
    spec = env.spec
    assert env.action_space.n == len(spec.shifts_K)
    assert env.observation_space.shape == (spec.observation_size,)
    env.reset(seed=3)
    actions = [4, 6, 8, 2, 0, 5] * 24  # indices of -4..+4 K
    rewards = []
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        assert info['hk_shift_K'] == spec.shifts_K[action]
        assert info['requested_power_fraction'] is None
        assert obs[7] == pytest.approx(spec.shifts_K[action] / 4)
        assert truncated == (i == len(actions) - 1)
    with patch('control.rl_offset.RLOffsetController') as controller:
        controller.return_value.shift_K.side_effect = [spec.shifts_K[a] for a in actions]
        result = run_simulation(config, weather_points=weather[:-1])
    metrics = calculate_metrics(result, config['building']['comfort_buffer_K'],
                                config['simulation']['dt_minutes'] / 60, config['building']['T_set_C'])
    score = calculate_score(result, metrics, config['simulation']['dt_minutes'] / 60,
                            ScoringParams.from_config(config['scoring']))
    assert -sum(rewards) == pytest.approx(score.total_score)


def test_power_environment_is_unchanged_by_the_offset_option():
    config, weather = setup()
    config['control']['method'] = 'heating_curve'
    env = HeatPumpEnv(config, weather, random_start=False)
    assert not env.offset_mode and env.action_space.n == len(env.power_levels)


def test_training_saves_loadable_offset_model_and_checks_the_contract(tmp_path):
    config, weather = setup()
    config['control']['reinforcement_learning_offset']['model_path'] = str(tmp_path / 'model.zip')
    settings = {'agent': 'offset', 'timesteps': 32, 'episode_days': .1, 'seed': 1, 'eval_every': 16,
                'training_period': (weather[0].timestamp, weather[-1].timestamp + timedelta(minutes=10)),
                'validation_period': (weather[24].timestamp, weather[-1].timestamp + timedelta(minutes=10)),
                'dqn': {'net_arch': [16, 16], 'buffer_size': 1000, 'batch_size': 8}}
    with patch('learning.training.load_weather_csv', return_value=weather):
        report = train(config, settings)
    metadata = report['metadata']
    assert metadata['agent'] == 'offset' and metadata['observation_version'] == OFFSET_OBSERVATION_VERSION
    assert metadata['shifts_K'][0] == -4.0 and metadata['shifts_K'][-1] == 4.0 and 'power_levels' not in metadata
    controller = RLOffsetController(config)
    assert controller.model.action_space.n == 9
    env = HeatPumpEnv(config, weather, random_start=False)
    obs, _ = env.reset()
    point = weather[0]
    with patch.object(controller.model, 'predict', return_value=(7, None)) as predict:
        shift = controller.shift_K(env.state, SimulationInput(point.outside_temperature_C, point.timestamp))
    assert shift == 3.0
    assert predict.call_args.args[0] == pytest.approx(obs)
    archived = list(tmp_path.glob('training_*/inference.yaml'))
    assert len(archived) == 1
    assert 'reinforcement_learning_offset' in archived[0].read_text()
    # The engine runs the trained controller with the plain heating curve for settling and power
    result = run_simulation(config, weather_points=weather[:-1])
    assert len(result.room_temperatures_C) == len(weather) - 1
    assert {row['hk_shift_K'] for row in result.hydraulics} <= set(controller.spec.shifts_K)
    # A model trained with another action list or observation layout is rejected
    config['control']['reinforcement_learning_offset']['shift_K'] = [-1, 0, 1]
    with pytest.raises(ValueError, match='contract'):
        RLOffsetController(config)
    config['control']['reinforcement_learning_offset']['shift_K'] = {'max': 4, 'step': 1}
    config['control']['reinforcement_learning_offset']['observation'] = {'last_shift': False}
    with pytest.raises(ValueError, match='contract'):
        RLOffsetController(config)
    metadata_path = tmp_path / 'model.json'
    stored = json.loads(metadata_path.read_text())
    assert stored['agent'] == 'offset'


def test_offset_training_settings_file_selects_the_offset_method():
    root = Path(__file__).resolve().parents[1]
    config, settings = load_training_settings(root / 'config/rl_training_offset.yaml')
    assert settings['agent'] == 'offset'
    assert config['control']['method'] == 'reinforcement_learning_offset'
    config, settings = load_training_settings(root / 'config/rl_training.yaml')
    assert settings['agent'] == 'power' and config['control']['method'] != 'reinforcement_learning_offset'
