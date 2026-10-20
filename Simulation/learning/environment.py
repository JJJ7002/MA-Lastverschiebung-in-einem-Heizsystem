"""Gymnasium adapter over the same physics and constraints used for inference."""
from dataclasses import asdict
from copy import deepcopy
from dataclasses import replace
from math import ceil
from math import isfinite

import gymnasium as gym
import numpy as np

from analysis.scoring import ScoringParams
from weather.loader import resample_weather
from control.rl_power import observation, power_levels_from_config, OBSERVATION_SIZE
from control.rl_offset import OffsetSpec, offset_observation
from simulation.engine import build_simulation_params, initial_state
from simulation.step import SimulationInput, simulate_step


class HeatPumpEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, config, weather, episode_steps=None, random_start=True, settling_weather=(), segments=None):
        super().__init__()
        self.config = deepcopy(config)
        # method reinforcement_learning_offset: the action is a heating-circuit set-point shift and the
        # heating curve sets the power; otherwise the action is the power level itself.
        self.offset_mode = self.config['control'].get('method') == 'reinforcement_learning_offset'
        if self.offset_mode:
            self.spec = OffsetSpec.from_config(self.config)
        else:
            self.config['control']['method'] = 'reinforcement_learning'
        self.params = build_simulation_params(self.config)
        dt_h = self.params.dt_hours
        if segments is not None:
            # Sequential evaluation windows: [(history, points), ...]; each window is settled on its own history.
            if random_start:
                raise ValueError('segments require random_start=False')
            combined, self.data_segments = [], []
            for history, points in segments:
                history = list(resample_weather(history, dt_h))
                if combined:
                    history = [p for p in history if p.timestamp > combined[-1].timestamp]
                combined += history
                points = list(resample_weather(points, dt_h))
                self.data_segments.append((len(combined), len(combined) + len(points) - 1))
                combined += points
            self.settling_weather = ()
            self.weather = tuple(combined)
            self.data_start = self.data_segments[0][0]
        else:
            self.weather = tuple(resample_weather(weather, dt_h))
            self.settling_weather = tuple(resample_weather(settling_weather, dt_h))
            self.data_start = len(self.settling_weather)
            self.weather = self.settling_weather + self.weather
        self.settling_steps = ceil(float(config['simulation'].get('settling_hours', 0)) / self.params.dt_hours)
        self.scoring = ScoringParams.from_config(config['scoring'])
        self.power_levels = power_levels_from_config(config)
        self.comfort_buffer = float(config['building']['comfort_buffer_K'])
        if not isfinite(self.comfort_buffer) or self.comfort_buffer < 0:
            raise ValueError('comfort_buffer_K must be finite and nonnegative')
        # One additional weather observation is needed for bootstrapping at truncation.
        if len(self.weather) - self.data_start < 2:
            raise ValueError('RL needs at least two weather samples')
        if segments is None and len(self.settling_weather) < self.settling_steps:
            raise ValueError('RL settling requires complete weather history before the episode period')
        dt = self.params.dt_hours
        if not isfinite(dt) or dt <= 0:
            raise ValueError('dt_hours must be finite and positive')
        step_s = dt * 3600
        # Weather may contain gaps (skipped seasonal shutdown); episodes never cross a gap.
        self.run_start, self.run_end = [], []
        for i, point in enumerate(self.weather):
            if i and abs((point.timestamp - self.weather[i - 1].timestamp).total_seconds() - step_s) <= .01:
                self.run_start.append(self.run_start[-1])
            else:
                if i and (point.timestamp - self.weather[i - 1].timestamp).total_seconds() <= 0:
                    raise ValueError('RL weather must be contiguous and match simulation.dt_minutes')
                self.run_start.append(i)
        last = len(self.weather) - 1
        for i in range(last, -1, -1):
            same = i < last and self.run_start[i + 1] == self.run_start[i]
            self.run_end.insert(0, self.run_end[0] if same else i)
        available = len(self.weather) - self.data_start
        self.random_start = random_start
        if segments is None:
            self.data_segments = [(self.data_start, self.run_end[self.data_start])] + [
                (i, self.run_end[i]) for i in range(self.data_start + 1, len(self.weather)) if self.run_start[i] == i]
        if episode_steps is None and not random_start:
            # Sequential pass: every heating run from its first to its last sample.
            self.episode_steps = sum(end - start for start, end in self.data_segments)
        else:
            self.episode_steps = available - 1 if episode_steps is None else int(episode_steps)
        if self.episode_steps < 1 or (random_start and self.episode_steps >= available):
            raise ValueError('episode_steps must be positive and smaller than the weather sample count')
        if random_start:
            self.valid_starts = [i for i in range(self.data_start, len(self.weather))
                                 if self.run_end[i] >= i + self.episode_steps]
        else:
            self.valid_starts = [self.data_start]
        self.segment_index = 0
        if not self.valid_starts:
            raise ValueError('No heating period is long enough for one episode')
        self.action_space = gym.spaces.Discrete(len(self.spec.shifts_K) if self.offset_mode else len(self.power_levels))
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(self.spec.observation_size if self.offset_mode else OBSERVATION_SIZE,),
            dtype=np.float32)
        self.finished = True

    def _observation(self):
        point = self.weather[self.index]
        if self.offset_mode:
            return np.asarray(offset_observation(self.state, point.outside_temperature_C, point.timestamp,
                                                 self.params, self.spec), dtype=np.float32)
        return np.asarray(observation(self.state, point.outside_temperature_C, point.timestamp, self.params),
                          dtype=np.float32)

    def _begin_run(self, index):
        self.index = index
        warmup_start = max(index - self.settling_steps, self.run_start[index])
        self.state = initial_state(self.config, self.weather[warmup_start].outside_temperature_C)
        settling_params = replace(self.params, heating_curve_control=True, cop_control=None)
        for point in self.weather[warmup_start:index]:
            result = simulate_step(self.state, SimulationInput(point.outside_temperature_C, point.timestamp), settling_params)
            self.state = result.state
        self.state = replace(self.state, total_electric_energy_Wh=0., total_thermal_energy_Wh=0., steps_elapsed=0,
                             buffer_target_C=None, requested_thermal_power_W=None,
                             building_loss_feedforward_W=None, buffer_temperature_correction_W=None,
                             shutdown_check_heat_W=None, minimum_heat_W=None)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._begin_run(int(self.valid_starts[int(self.np_random.integers(0, len(self.valid_starts)))])
                        if self.random_start else self.valid_starts[0])
        self.steps = 0
        self.segment_index = 0
        self.finished = False
        return self._observation(), {}

    def step(self, action):
        if self.finished:
            raise RuntimeError('Call reset() before stepping a new episode')
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid RL action: {action}')
        shift = self.spec.shifts_K[int(action)] if self.offset_mode else None
        request = None if self.offset_mode else self.power_levels[int(action)]
        totals = None
        for _ in range(min(self.params.control_steps, self.episode_steps - self.steps)):
            if (not self.random_start and self.index == self.data_segments[self.segment_index][1]
                    and self.segment_index + 1 < len(self.data_segments)):
                # Seasonal pause, data gap or next evaluation window: continue with a fresh start state.
                self.segment_index += 1
                self._begin_run(self.data_segments[self.segment_index][0])
            point = self.weather[self.index]
            previous_on = self.state.heat_pump.is_on
            if self.offset_mode:
                self.state = replace(self.state, hk_shift_K=shift)  # held until the next decision
            result = simulate_step(self.state, SimulationInput(point.outside_temperature_C, point.timestamp),
                                   self.params, requested_power_fraction=request)
            self.state = result.state
            threshold = (self.params.target_room_temperature_C - self.comfort_buffer
                         - self.params.operating.night_setback.room_reduction_K(point.timestamp, comfort=True))
            score = self.scoring.transition_score(
                point.timestamp, self.params.dt_hours, result.electric_power_W, previous_on,
                self.state.heat_pump.is_on, self.state.building.T_room_C, threshold,
                start_count=result.heat_pump_start_count)
            values = asdict(score)
            totals = values if totals is None else {key: value if key == 'currency' else totals[key] + value
                                                   for key, value in values.items()}
            self.steps += 1
            self.index += 1
        self.finished = self.steps >= self.episode_steps
        info = dict(totals)
        info.update(requested_power_fraction=request, hk_shift_K=shift, power_fraction=result.power_fraction,
                    room_temperature_C=self.state.building.T_room_C,
                    mass_temperature_C=self.state.building.T_mass_C,
                    storage_temperature_C=self.state.storage.control_C,
                    storage_mean_temperature_C=self.state.storage.mean_temperature_C)
        # A sampled weather interval is a time limit, not the end of heating demand.
        # SB3 bootstraps from this next observation instead of learning free final cooling.
        return self._observation(), -totals['total_score'], False, self.finished, info
