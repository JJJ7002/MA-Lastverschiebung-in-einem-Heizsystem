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
from control.rl_power import RLController, OBSERVATION_VERSION, OBSERVATION_SIZE
from weather.loader import WeatherPoint
from learning.environment import HeatPumpEnv
from simulation.engine import run_simulation
from simulation.step import SimulationInput
from analysis.metrics import calculate_metrics
from analysis.scoring import calculate_score, ScoringParams
from learning.training import train


def setup():
    config = load_test_config()
    config['weather']['source'] = 'csv'  # Isolate model tests from network access
    config['simulation']['weather_file'] = 'data/weather/ogd-smn_bas_t_historical_2020-2029.csv'
    config['simulation']['dt_minutes'] = 10
    config['simulation'].update(start_time=None, end_time=None, settling_hours=0)
    config['scoring']['electricity']['tariff_windows_enabled'] = True
    weather = [WeatherPoint(datetime(2020, 1, 1, 20) + timedelta(minutes=10*i), -5)
               for i in range(145)]
    return config, weather


def test_reward_matches_existing_score_and_inference():
    config, weather = setup()
    env = HeatPumpEnv(config, weather, random_start=False)
    check_env(env)
    env.reset(seed=5)
    actions = [0, 1, 2, 3, 4, 0] * 24
    rewards, powers = [], []
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        powers.append(info['power_fraction'])
        assert obs[-1] == pytest.approx(info['power_fraction'])
        assert not terminated
        assert truncated == (i == len(actions) - 1)
    with pytest.raises(RuntimeError):
        env.step(0)
    config['control']['method'] = 'reinforcement_learning'
    with patch('control.rl_power.RLController') as controller:
        controller.return_value.predict.side_effect = [env.power_levels[a] for a in actions]
        result = run_simulation(config, weather_points=weather[:-1])
    metrics = calculate_metrics(result, config['building']['comfort_buffer_K'],
                                (config['simulation']['dt_minutes'] / 60), config['building']['T_set_C'])
    score = calculate_score(result, metrics, (config['simulation']['dt_minutes'] / 60),
                            ScoringParams.from_config(config['scoring']))
    assert -sum(rewards) == pytest.approx(score.total_score)
    assert powers == result.heat_pump_power_fractions
    assert env.state.building.T_room_C == result.room_temperatures_C[-1]
    assert env.state.storage.temperature_C == result.storage_temperatures_C[-1]


def test_gap_without_long_enough_run_and_invalid_actions_rejected():
    config, weather = setup()
    with pytest.raises(ValueError, match='No heating period'):
        HeatPumpEnv(config, weather[:2] + weather[3:])
    env = HeatPumpEnv(config, weather)
    env.reset()
    with pytest.raises(ValueError):
        env.step(len(env.power_levels))


def test_training_save_reload_and_periods(tmp_path):
    config, weather = setup()
    config['control']['reinforcement_learning']['model_path'] = str(tmp_path / 'model.zip')
    settings = {'timesteps': 32, 'episode_days': .1, 'seed': 1, 'eval_every': 16,
                'training_period': (weather[0].timestamp, weather[-1].timestamp + timedelta(minutes=10)),
                'validation_period': (weather[24].timestamp, weather[-1].timestamp + timedelta(minutes=10)),
                'dqn': {'net_arch': [16, 16], 'buffer_size': 1000, 'batch_size': 8}, 'validation_windows': 2, 'validation_window_days': 0.1}
    with patch('learning.training.load_weather_csv', return_value=weather):
        report = train(config, settings)
    periods = report['metadata']['periods']
    assert periods['train']['samples'] == len(weather)
    assert report['metadata']['validation_windows'] == 2 and periods['validation']['samples'] == 28
    assert 'test_comparison' not in report
    controller = RLController(config)
    assert controller.model.observation_space.shape == (OBSERVATION_SIZE,)
    env = HeatPumpEnv(config, weather, random_start=False)
    obs, _ = env.reset()
    with patch.object(controller.model, 'predict', return_value=(0, None)) as predict:
        point = weather[0]
        controller.predict(env.state, SimulationInput(point.outside_temperature_C, point.timestamp))
        assert predict.call_args.args[0] == pytest.approx(obs)
    metadata_path = tmp_path / 'model.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['observation_version'] = 1
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='contract'):
        RLController(config)
    metadata['observation_version'] = OBSERVATION_VERSION
    metadata['power_levels'] = [0, .5, .6, .8, 1]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='contract'):
        RLController(config)


def _gap_weather():
    # 30 days, the 2nd Sep..: two heating runs separated by a skipped gap (like a summer pause)
    first = [WeatherPoint(datetime(2024, 5, 20) + timedelta(minutes=10 * i), 5) for i in range(144 * 5)]
    second = [WeatherPoint(datetime(2024, 9, 1) + timedelta(minutes=10 * i), 5) for i in range(144 * 5)]
    return first + second


def test_env_episodes_never_cross_a_skipped_gap():
    config, _ = setup()
    weather = _gap_weather()
    env = HeatPumpEnv(config, weather, episode_steps=144 * 2, random_start=True)
    boundary = 144 * 5
    for seed in range(20):
        env.reset(seed=seed)
        assert (env.index + env.episode_steps < boundary) or env.index >= boundary
    assert env.valid_starts


def test_env_sequential_pass_restarts_after_gap():
    config, _ = setup()
    weather = _gap_weather()
    env = HeatPumpEnv(config, weather, random_start=False)
    assert env.episode_steps == 2 * (144 * 5 - 1)
    env.reset(seed=0)
    jumped = False
    for _ in range(env.episode_steps):
        before = env.index
        _, _, _, truncated, _ = env.step(0)
        jumped |= env.index > before + 1 and env.weather[env.index].timestamp.month == 9
    assert truncated
    assert env.weather[env.index].timestamp == datetime(2024, 9, 1) + timedelta(minutes=10 * (144 * 5 - 1))


def test_env_without_gap_is_unchanged():
    config, weather = setup()
    env = HeatPumpEnv(config, weather, random_start=False)
    assert env.episode_steps == len(weather) - 1
    assert env.valid_starts == [0]


def test_training_settings_file(tmp_path):
    from learning.training import load_training_settings
    (tmp_path / 'plant.yaml').write_text((Path(__file__).resolve().parents[1] / 'config/default.yaml').read_text())
    (tmp_path / 'rl.yaml').write_text("""plant_config: plant.yaml
training: {timesteps: 100, seed: 3}
periods:
  training: {start_time: "2024-01-01 00:00", end_time: "2024-03-01 00:00"}
""")
    config, settings = load_training_settings(tmp_path / 'rl.yaml')
    assert settings['timesteps'] == 100 and settings['episode_days'] == 14 and settings['seed'] == 3
    assert settings['validation_period'] == settings['training_period']
    (tmp_path / 'bad.yaml').write_text("plant_config: plant.yaml\nperiods: {training: {start_time: '2024-01-01 00:00'}}\n")
    with pytest.raises(ValueError, match='periods.training'):
        load_training_settings(tmp_path / 'bad.yaml')


def test_measurement_gap_is_skipped_like_a_pause_in_training_slices():
    from learning.training import _slice
    config, _ = setup()
    from simulation.engine import build_simulation_params
    schedule = build_simulation_params(config).operating.schedule
    base = datetime(2023, 1, 15)
    points = [WeatherPoint(base + timedelta(minutes=10 * i), 0) for i in range(10)]
    points += [WeatherPoint(base + timedelta(hours=40, minutes=10 * i), 0) for i in range(10)]
    data, history = _slice(points, schedule, (points[12].timestamp, points[-1].timestamp), 2, 600)
    assert [p.timestamp for p in history] == [points[10].timestamp, points[11].timestamp]
    with pytest.raises(ValueError, match='Insufficient'):
        _slice(points, schedule, (points[11].timestamp, points[-1].timestamp), 3, 600)


def test_dqn_settings_defaults_overrides_and_validation(tmp_path):
    from learning.training import dqn_settings, DQN_DEFAULTS
    assert dqn_settings(None) == DQN_DEFAULTS
    assert dqn_settings({'net_arch': [128, 128], 'gamma': .99})['net_arch'] == [128, 128]
    with pytest.raises(ValueError, match='Unknown'):
        dqn_settings({'layers': [8]})
    with pytest.raises(ValueError, match='Invalid'):
        dqn_settings({'gamma': 1.5})


def test_validation_windows_are_spread_settled_and_restart():
    from learning.training import sample_windows, history_before
    config, _ = setup()
    config['simulation']['settling_hours'] = 2
    base = datetime(2024, 1, 1)
    raw = [WeatherPoint(base + timedelta(minutes=10 * i), 2.0) for i in range(144 * 30)]
    windows = sample_windows(raw, 3, 144 * 2, 600)
    assert len(windows) == 3 and all(len(w) == 144 * 2 for w in windows)
    starts = [w[0].timestamp for w in windows]
    assert starts == sorted(starts) and windows[0][-1].timestamp < windows[1][0].timestamp
    index = {p.timestamp: i for i, p in enumerate(raw)}
    segments = [(history_before(raw, index, w[0].timestamp, 12, 600), w) for w in windows]
    assert all(len(h) == 12 for h, _ in segments)
    env = HeatPumpEnv(config, (), random_start=False, segments=segments)
    assert env.episode_steps == 3 * (144 * 2 - 1)
    env.reset(seed=0)
    firsts = [env.index]
    for _ in range(env.episode_steps):
        _, _, _, truncated, _ = env.step(0)
        if env.segment_index + 1 > len(firsts):
            firsts.append(env.index)
    assert truncated and len(firsts) == 3
    assert env.weather[env.data_segments[1][0]].timestamp == windows[1][0].timestamp
    short = sample_windows(raw[:100], 3, 144, 600)  # fewer samples than requested: whole data
    assert len(short) == 1 and len(short[0]) == 100
