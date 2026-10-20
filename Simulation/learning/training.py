"""Train a DQN controller. All settings are read from a YAML file (default: config/rl_training.yaml):
training and validation period, number of decisions, episode length and seed. Both periods are
independent of the simulation period of the plant configuration. The checkpoint with the best
validation score is always adopted as the model; there is no separate test run.
``agent: offset`` in the YAML trains the offset controller (control.method reinforcement_learning_offset):
the agent shifts the heating-circuit set-point and the heating curve sets the power."""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta
import json
from math import ceil
import shutil
from pathlib import Path

import yaml

from simulation.config import load_config
from control.rl_power import (OBSERVATION_VERSION, model_path_from_config,
                                            power_levels_from_config)
from control.rl_offset import OffsetSpec, offset_model_path_from_config
from weather.loader import load_weather_csv, resample_weather
from weather.meteoswiss import weather_source, load_api_weather
from simulation.engine import _parse_datetime, build_simulation_params

DEFAULT_SETTINGS = Path(__file__).resolve().parents[1] / 'config/rl_training.yaml'
DQN_DEFAULTS = {'net_arch': [64, 64], 'learning_rate': 1e-4, 'buffer_size': 100_000, 'learning_starts': 1000,
                'batch_size': 64, 'gamma': 0.995, 'train_freq': 4, 'gradient_steps': 1,
                'target_update_interval': 5000, 'exploration_fraction': 0.3,
                'exploration_initial_eps': 1.0, 'exploration_final_eps': 0.05}


def dqn_settings(raw):
    """Merge the optional ``dqn`` section with the defaults and check the values."""
    unknown = set(raw or {}) - set(DQN_DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown dqn setting(s): {", ".join(sorted(unknown))}')
    values = {**DQN_DEFAULTS, **(raw or {})}
    values['net_arch'] = [int(n) for n in values['net_arch']]
    for name in ('buffer_size', 'learning_starts', 'batch_size', 'train_freq', 'gradient_steps',
                 'target_update_interval'):
        values[name] = int(values[name])
    for name in ('learning_rate', 'gamma', 'exploration_fraction', 'exploration_initial_eps',
                 'exploration_final_eps'):
        values[name] = float(values[name])
    if (not values['net_arch'] or min(values['net_arch']) < 1 or values['learning_rate'] <= 0
            or not 0 < values['gamma'] <= 1 or not 0 <= values['exploration_fraction'] <= 1
            or not 0 <= values['exploration_final_eps'] <= values['exploration_initial_eps'] <= 1
            or min(values['buffer_size'], values['batch_size'], values['train_freq'],
                   values['gradient_steps'], values['target_update_interval']) < 1
            or values['learning_starts'] < 0):
        raise ValueError('Invalid dqn settings')
    return values


def load_training_settings(path):
    """Return (plant configuration, validated settings) from a training YAML file."""
    path = Path(path)
    raw = load_config(path)
    plant = path.parent / str(raw.get('plant_config', 'default.yaml'))
    config = load_config(plant)
    agent = str(raw.get('agent', 'power'))
    if agent not in ('power', 'offset'):
        raise ValueError("agent must be 'power' (agent chooses the power level) or 'offset' "
                         "(agent shifts the heating-circuit set-point)")
    if agent == 'offset':
        config['control']['method'] = 'reinforcement_learning_offset'
    training = dict(raw.get('training') or {})
    periods = dict(raw.get('periods') or {})
    train_period = dict(periods.get('training') or {})
    validation_period = dict(periods.get('validation') or train_period)

    def window(name, period):
        start, end = _parse_datetime(period.get('start_time')), _parse_datetime(period.get('end_time'))
        if start is None or end is None or end <= start:
            raise ValueError(f'periods.{name}: start_time and end_time are required and end must follow start')
        return start, end

    settings = {'agent': agent,
                'timesteps': int(training.get('timesteps', 200_000)),
                'episode_days': float(training.get('episode_days', 14)),
                'seed': int(training.get('seed', 42)),
                'eval_every': int(training.get('eval_every', 25_000)),
                'training_period': window('training', train_period),
                'validation_period': window('validation', validation_period),
                'n_envs': int(training.get('n_envs', 1)),
                'validation_windows': training.get('validation_windows'),
                'validation_window_days': float(training.get('validation_window_days', 7)),
                'dqn': dqn_settings(raw.get('dqn'))}
    if settings['n_envs'] < 1 or settings['validation_window_days'] <= 0:
        raise ValueError('training.n_envs and validation_window_days must be positive')
    if settings['timesteps'] <= 0 or settings['episode_days'] <= 0 or settings['eval_every'] <= 0:
        raise ValueError('training.timesteps, episode_days and eval_every must be positive')
    return config, settings


def _slice(raw_weather, schedule, period, settling_count, step_s):
    """Heating-season points of a period (seasonal pause removed) and the weather history before it."""
    start, end = period
    points = [p for p in raw_weather if start <= p.timestamp < end and not schedule.heating_off_at(p.timestamp)]
    history = []
    if settling_count:
        earlier = [p for p in raw_weather if p.timestamp < start]
        for point in reversed(earlier):  # settling history must be contiguous up to the period start
            if len(history) == settling_count or (
                    history and (history[0].timestamp - point.timestamp).total_seconds() > step_s * 1.01):
                break
            history.insert(0, point)
    if len(history) != settling_count:
        raise ValueError(f'Insufficient weather data before {start} for RL settling')
    if len(points) < 2:
        raise ValueError(f'Period {start} .. {end} contains too little weather data')
    return points, history


def sample_windows(points, count, length, step_s):
    """``count`` evenly spread, non-overlapping windows of ``length`` samples that never cross a data gap."""
    runs, current = [], []
    for point in points:
        if current and abs((point.timestamp - current[-1].timestamp).total_seconds() - step_s) > .01:
            runs.append(current)
            current = []
        current.append(point)
    if current:
        runs.append(current)
    total = len(points)
    if count * length >= total:
        return runs
    offsets = [sum(len(r) for r in runs[:i]) for i in range(len(runs))]
    windows, previous_end = [], {}
    for k in range(count):
        position = int((k + 0.5) * total / count)
        i = max(j for j in range(len(runs)) if offsets[j] <= position)
        run = runs[i]
        if len(run) <= length:
            begin, end = 0, len(run)
        else:
            begin = min(max(position - offsets[i] - length // 2, 0), len(run) - length)
            begin = max(begin, previous_end.get(i, 0))
            end = begin + length
            if end > len(run):
                continue
        if len(run) <= length and i in previous_end:
            continue
        previous_end[i] = end
        windows.append(run[begin:end])
    return windows


def history_before(raw_weather, index_by_time, first, settling_count, step_s):
    """Up to ``settling_count`` contiguous raw samples directly before timestamp ``first``."""
    if not settling_count or first not in index_by_time:
        return []
    end = index_by_time[first]
    history = []
    for point in reversed(raw_weather[max(0, end - settling_count):end]):
        if history and (history[0].timestamp - point.timestamp).total_seconds() > step_s * 1.01:
            break
        history.insert(0, point)
    return history


def _make_training_env(config, weather, episode_steps, history):
    from learning.environment import HeatPumpEnv
    return HeatPumpEnv(config, weather, episode_steps, settling_weather=history)


def train(config, settings):
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    import torch
    from learning.environment import HeatPumpEnv

    timesteps, seed, eval_every = settings['timesteps'], settings['seed'], settings['eval_every']
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    dt_hours = float(config['simulation']['dt_minutes']) / 60
    settling_count = ceil(float(config['simulation'].get('settling_hours', 0)) / dt_hours)
    periods = {'train': settings['training_period'], 'validation': settings['validation_period']}
    first = min(start for start, _ in periods.values())
    last = max(end for _, end in periods.values())
    weather_config = deepcopy(config)
    weather_config['simulation']['start_time'] = first - timedelta(hours=settling_count * dt_hours)
    weather_config['simulation']['end_time'] = last
    weather = (load_api_weather(weather_config, root, allow_gaps=True) if weather_source(config) == 'meteoswiss'
               else load_weather_csv(root / config['simulation']['weather_file']))
    weather = resample_weather(weather, dt_hours)
    schedule = build_simulation_params(config).operating.schedule  # seasonal pauses and measurement gaps are skipped
    partitions, histories = {}, {}
    for name, period in periods.items():
        partitions[name], histories[name] = _slice(weather, schedule, period, settling_count, dt_hours * 3600)
    longest, run = 0, 0
    for index, point in enumerate(partitions['train']):
        contiguous = index and abs((point.timestamp - partitions['train'][index - 1].timestamp).total_seconds()
                                   - dt_hours * 3600) <= .01
        run = run + 1 if contiguous else 1
        longest = max(longest, run)
    episode_steps = min(max(1, round(settings['episode_days'] * 24 / dt_hours)), longest - 1)
    n_envs = settings.get('n_envs', 1)
    env_kwargs = dict(config=config, weather=partitions['train'], episode_steps=episode_steps,
                      history=histories['train'])
    probe = HeatPumpEnv(config, partitions['train'], episode_steps, settling_weather=histories['train'])
    check_env(probe)
    if n_envs > 1:
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv
        env = make_vec_env(_make_training_env, n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv,
                           env_kwargs=env_kwargs)
    else:
        env = Monitor(probe)
    windows = settings.get('validation_windows')
    if windows:
        length = max(2, round(float(settings.get('validation_window_days', 7)) * 24 / dt_hours))
        step_s = dt_hours * 3600
        sampled = sample_windows(partitions['validation'], int(windows), length, step_s)
        index_by_time = {p.timestamp: i for i, p in enumerate(weather)}
        segments = [(history_before(weather, index_by_time, w[0].timestamp, settling_count, step_s), w)
                    for w in sampled]
        validation = Monitor(HeatPumpEnv(config, (), random_start=False, segments=segments))
        partitions['validation'] = [p for w in sampled for p in w]
    else:
        validation = Monitor(HeatPumpEnv(config, partitions['validation'], random_start=False,
                                         settling_weather=histories['validation']))
    offset = config['control'].get('method') == 'reinforcement_learning_offset'
    path = offset_model_path_from_config(config) if offset else model_path_from_config(config)
    run_dir = path.parent / ('training_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run_dir.mkdir(parents=True)
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    (run_dir / 'rl_training.yaml').write_text(yaml.safe_dump(
        {**settings, 'training_period': [str(t) for t in settings['training_period']],
         'validation_period': [str(t) for t in settings['validation_period']]}, sort_keys=False),
        encoding='utf-8')
    eval_freq = max(1, min(eval_every, timesteps) // n_envs)
    callback = EvalCallback(validation, best_model_save_path=str(run_dir),
                            log_path=str(run_dir), eval_freq=eval_freq,
                            n_eval_episodes=1, deterministic=True)
    hp = dqn_settings(settings.get('dqn'))
    # train_freq and gradient_steps are given per decision; with several environments one
    # vectorised step delivers n_envs decisions, so the ratio of updates to decisions is kept.
    vec_freq = max(1, round(hp['train_freq'] / n_envs))
    vec_gradient_steps = max(1, round(hp['gradient_steps'] * vec_freq * n_envs / hp['train_freq']))
    model = DQN('MlpPolicy', env, learning_rate=hp['learning_rate'], buffer_size=hp['buffer_size'],
                learning_starts=min(hp['learning_starts'], max(0, timesteps // 10)),
                batch_size=hp['batch_size'], gamma=hp['gamma'], train_freq=vec_freq,
                gradient_steps=vec_gradient_steps, target_update_interval=max(1, hp['target_update_interval'] // n_envs),
                exploration_fraction=hp['exploration_fraction'],
                exploration_initial_eps=hp['exploration_initial_eps'],
                exploration_final_eps=hp['exploration_final_eps'],
                policy_kwargs={'net_arch': hp['net_arch']}, seed=seed, device='cpu', verbose=1)
    model.learn(total_timesteps=timesteps, callback=callback)
    training_timesteps = int(model.num_timesteps)
    if (model.num_timesteps % (eval_freq * n_envs) == 0 and callback.last_mean_reward != float('-inf')
            and callback.last_mean_reward == callback.last_mean_reward):
        final_reward = float(callback.last_mean_reward)  # the last validation run already rated the final model
    else:
        final_reward, _ = evaluate_policy(model, validation, n_eval_episodes=1, deterministic=True)
    if (run_dir / 'best_model.zip').exists() and callback.best_mean_reward >= final_reward:
        model = DQN.load(run_dir / 'best_model.zip', device='cpu')
        validation_reward = float(callback.best_mean_reward)
    else:
        validation_reward = float(final_reward)
    model.save(path)
    contract = ({'agent': 'offset', **OffsetSpec.from_config(config).metadata()} if offset
                else {'observation_version': OBSERVATION_VERSION,
                      'power_levels': list(power_levels_from_config(config))})
    metadata = {**contract,
                'seed': seed, 'dqn': hp, 'n_envs': n_envs, 'validation_windows': len(sampled) if windows else None, 'training_timesteps': training_timesteps,
                'selected_checkpoint_timesteps': int(model.num_timesteps),
                'validation_total_score': -validation_reward,
                'periods': {name: {'start': start.isoformat(sep=' '), 'end': end.isoformat(sep=' '),
                                   'samples': len(partitions[name])}
                            for name, (start, end) in periods.items()}}
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    # Keep each inference configuration reproducible after another training run.
    archived_model = run_dir / 'selected_model.zip'
    shutil.copyfile(path, archived_model)
    shutil.copyfile(path.with_suffix('.json'), archived_model.with_suffix('.json'))
    report = {'model': str(path), 'metadata': metadata}
    (run_dir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    inference_config = deepcopy(config)
    section = 'reinforcement_learning_offset' if offset else 'reinforcement_learning'
    inference_config['control']['method'] = section
    inference_config['control'].setdefault(section, {})['model_path'] = str(archived_model)
    inference_config['analysis']['output_dir'] = str(run_dir / 'simulation')
    (run_dir / 'inference.yaml').write_text(yaml.safe_dump(inference_config, sort_keys=False), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'Run saved model: python main.py --config "{run_dir / "inference.yaml"}"')
    env.close()
    validation.close()
    probe.close()
    return report


def main(default_settings=DEFAULT_SETTINGS):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=default_settings,
                        help=f'training settings (default: {default_settings.relative_to(DEFAULT_SETTINGS.parent.parent)})')
    args = parser.parse_args()
    config, settings = load_training_settings(args.config)
    train(config, settings)


if __name__ == '__main__':
    main()
