"""Observation contract and optional DQN inference; no ML imports for other controllers."""
from datetime import datetime, timedelta
from math import cos, sin, pi, isfinite
from pathlib import Path
import json
import sys

from control.operating_constraints import OperatingState
from models.heat_pump import quantize_power_command, validate_power_command
from models.performance_map import load_performance_map


OBSERVATION_VERSION = 15
OBSERVATION_SIZE = 10
OFF_POWER_LEVEL = -1.0
TRACKING_SCALE_K = 0.5  # room-temperature deviation unit: fine resolution around the target


def observation(state, outside_C, timestamp, params):
    """Features of one control decision, all of order one.

    outside/20, room minus the current room target (0.5 K units, clipped to +-8), sine and
    cosine of the time of day, storage top /
    middle / bottom temperature, the margin of the top layer above the supply temperature the
    heating circuit currently requires (10 K units), the room target itself (target - 20 K,
    night setback and warm-up ramp included) and the previous heat-pump power (-1 = off, 0 = minimum ... 1 = maximum).
    """
    room_C = state.building.T_room_C
    target_now_C = params.room_target_C_at(timestamp)
    nodes = state.storage.node_temperatures_C
    if nodes is None:
        nodes = (state.storage.temperature_C,) * params.storage.node_count
    top_C, middle_C = nodes[params.storage.heating_supply_node], nodes[len(nodes) // 2]
    bottom_C = nodes[params.storage.wp_return_node]
    demand_W = params.room_heat_demand_W(room_C, outside_C, target_now_C, state.building.T_mass_C)
    required_C = params.heating_flow(top_C, room_C, demand_W, outside_C, timestamp).target_supply_C
    minutes = timestamp.hour * 60 + timestamp.minute + timestamp.second / 60
    angle = 2 * pi * minutes / 1440
    values = (outside_C / 20,
              min(8.0, max(-8.0, (room_C - target_now_C) / TRACKING_SCALE_K)),
              sin(angle), cos(angle),
              (top_C - 45) / 20, (middle_C - 40) / 20, (bottom_C - 35) / 20,
              min(3.0, max(-3.0, (top_C - required_C) / 10)),
              target_now_C - 20.0,
              state.power_fraction)
    if not all(isfinite(value) for value in values):
        raise ValueError('RL observations must be finite')
    return values


def power_levels_from_config(config):
    map_path = (Path(__file__).resolve().parents[1] / config['heat_pump']['performance_map_file']).resolve()
    levels = load_performance_map(str(map_path)).power_fractions
    values = (OFF_POWER_LEVEL, *levels)
    configured = config['control'].get('reinforcement_learning', {}).get('power_levels')
    if configured is not None and tuple(float(x) for x in configured) != values:
        raise ValueError('RL power_levels are defined by heat_pump.performance_map_file; remove the custom list')
    return values


def apply_power_request(requested, storage_C, state, control, heat_pump,
                        step_minutes, timestamp: datetime | None):
    """Apply shared equipment constraints and minimum run/idle times."""
    validate_power_command(requested)
    requested = quantize_power_command(requested, heat_pump)
    elapsed = state.minutes_in_current_state + step_minutes
    mode = control.schedule.mode_at(timestamp)
    if mode == 'force_on':
        requested = 1.0
    if mode == 'force_off' or storage_C >= heat_pump.sink_temp_max_C:
        requested = -1.0
        on = False
    elif state.is_on:
        on = requested >= 0 or elapsed < heat_pump.min_on_time_min
    else:
        on = requested >= 0 and elapsed >= heat_pump.min_off_time_min
    power = max(0.0, requested) if on else -1.0
    return OperatingState(on, elapsed if on == state.is_on else 0), power


def model_path_from_config(config):
    value = config['control'].get('reinforcement_learning', {}).get(
        'model_path', 'results/rl/model.zip')
    path = (Path(__file__).resolve().parents[1] / value).resolve()
    return path if path.suffix == '.zip' else Path(str(path) + '.zip')


class RLController:
    def __init__(self, config):
        path = model_path_from_config(config)
        if not path.is_file():
            raise FileNotFoundError(f'RL model not found: {path}. Run train_rl.py from the project root first.')
        metadata_path = path.with_suffix('.json')
        if not metadata_path.is_file():
            raise FileNotFoundError(f'RL model metadata not found: {metadata_path}')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        self.power_levels = power_levels_from_config(config)
        from simulation.engine import build_simulation_params  # lazy: engine imports this module
        self.params = build_simulation_params(config)
        if (metadata.get('observation_version') != OBSERVATION_VERSION
                or tuple(metadata.get('power_levels', ())) != self.power_levels):
            raise ValueError('RL model observation/action contract differs from configuration; retrain the model')
        try:
            from stable_baselines3 import DQN
        except ImportError as exc:
            root = Path(__file__).resolve().parents[1]
            raise ImportError(
                f'RL dependencies unavailable in Python: {sys.executable}\n'
                f'Start with the project environment: "{root / "run_simulation.cmd"}"\n'
                f'Or install into this interpreter: "{sys.executable}" -m pip install '
                f'-r "{root / "requirements-rl.txt"}"'
            ) from exc
        self.model = DQN.load(path, device='cpu')
        if (self.model.observation_space.shape != (OBSERVATION_SIZE,)
                or self.model.action_space.n != len(self.power_levels)):
            raise ValueError('RL model has incompatible observation/action spaces')

    def predict(self, state, inputs):
        import numpy as np
        values = observation(state, inputs.outside_temperature_C, inputs.timestamp, self.params)
        action, _ = self.model.predict(np.asarray(values, dtype=np.float32), deterministic=True)
        return self.power_levels[int(action)]
