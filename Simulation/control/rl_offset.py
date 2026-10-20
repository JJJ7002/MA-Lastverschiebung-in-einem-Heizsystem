"""RL offset controller: a DQN chooses the heating-circuit set-point shift, the heating curve sets the power.

Like the real plant, the heat pump keeps its normal weather-compensated control (heating_curve).
The agent only moves the set-point of the heat pump heating circuit (WPM "WP HK") in kelvin; the
effect on the buffer target follows the measured curve gain (control.heating_curve.hk_shift_gain).
The observation contains only quantities a plant could measure, or the controller's own last output."""
from dataclasses import dataclass
from math import cos, sin, pi, isfinite
from pathlib import Path
import json
import sys

OFFSET_OBSERVATION_VERSION = 2
BASE_OBSERVATION_SIZE = 7
TRACKING_SCALE_K = 0.5  # room-temperature deviation unit: fine resolution around the target


@dataclass(frozen=True)
class OffsetSpec:
    """Action list (set-point shifts in K) and optional observation features."""
    shifts_K: tuple
    last_shift: bool = True
    heat_pump_power: bool = True

    def __post_init__(self):
        if len(self.shifts_K) < 2 or not all(isfinite(v) for v in self.shifts_K):
            raise ValueError('control.reinforcement_learning_offset.shift_K needs at least two finite values')
        if any(b <= a for a, b in zip(self.shifts_K, self.shifts_K[1:])):
            raise ValueError('control.reinforcement_learning_offset.shift_K must be strictly increasing')

    @property
    def observation_size(self):
        return BASE_OBSERVATION_SIZE + int(self.last_shift) + int(self.heat_pump_power)

    @classmethod
    def from_config(cls, config):
        settings = config['control'].get('reinforcement_learning_offset') or {}
        raw = settings.get('shift_K', {'max': 4.0, 'step': 1.0})
        if isinstance(raw, dict):
            # Symmetric range around 0: -max ... +max in steps of `step`
            limit, step = float(raw['max']), float(raw['step'])
            if not (isfinite(limit) and isfinite(step)) or step <= 0 or limit < step:
                raise ValueError('control.reinforcement_learning_offset.shift_K needs max >= step > 0')
            count = int(round(limit / step))
            if abs(count * step - limit) > 1e-9:
                raise ValueError('control.reinforcement_learning_offset.shift_K.max must be a multiple of step')
            values = tuple(round(k * step, 6) + 0.0 for k in range(-count, count + 1))
        else:
            values = tuple(float(v) for v in raw)
        features = settings.get('observation') or {}
        unknown = set(features) - {'last_shift', 'heat_pump_power'}
        if unknown:
            raise ValueError(f'Unknown control.reinforcement_learning_offset.observation entries: {sorted(unknown)}')
        for name, value in features.items():
            if not isinstance(value, bool):
                raise ValueError(f'control.reinforcement_learning_offset.observation.{name} must be a boolean')
        return cls(values, features.get('last_shift', True), features.get('heat_pump_power', True))

    def metadata(self):
        return {'observation_version': OFFSET_OBSERVATION_VERSION, 'shifts_K': list(self.shifts_K),
                'last_shift': self.last_shift, 'heat_pump_power': self.heat_pump_power}


def offset_observation(state, outside_C, timestamp, params, spec):
    """Measurable features of one control decision, all of order one.

    outside/20; room minus room target (0.5 K units, clipped to +-8); room target - 20 K (night
    setback and warm-up ramp included); sine and cosine of the time of day; storage sensor (T-40)/20;
    actual building supply after the mixer (T-35)/15; then optionally the last chosen shift
    (centred in the action range, -1..1) and the heat-pump power (-1 = off, 0 = minimum ... 1).
    The required supply set-point is not observed: in the model it follows from the building demand,
    which contains the unmeasurable mass temperature.
    """
    room_C = state.building.T_room_C
    target_C = params.room_target_C_at(timestamp)
    sensor_C = params.storage.control_temperature_C(state.storage)
    nodes = state.storage.node_temperatures_C
    top_C = nodes[params.storage.heating_supply_node] if nodes is not None else state.storage.temperature_C
    demand_W = params.room_heat_demand_W(room_C, outside_C, target_C, state.building.T_mass_C)
    flow = params.heating_flow(top_C, room_C, demand_W, outside_C, timestamp)
    minutes = timestamp.hour * 60 + timestamp.minute + timestamp.second / 60
    angle = 2 * pi * minutes / 1440
    values = [outside_C / 20,
              min(8.0, max(-8.0, (room_C - target_C) / TRACKING_SCALE_K)),
              target_C - 20.0,
              sin(angle), cos(angle),
              (sensor_C - 40) / 20,
              (flow.supply_C - 35) / 15]
    if spec.last_shift:
        low, high = spec.shifts_K[0], spec.shifts_K[-1]
        values.append((state.hk_shift_K - (low + high) / 2) / ((high - low) / 2))
    if spec.heat_pump_power:
        values.append(state.power_fraction)
    if not all(isfinite(value) for value in values):
        raise ValueError('RL observations must be finite')
    return tuple(values)


def offset_model_path_from_config(config):
    value = (config['control'].get('reinforcement_learning_offset') or {}).get(
        'model_path', 'results/rl_offset/model.zip')
    path = (Path(__file__).resolve().parents[1] / value).resolve()
    return path if path.suffix == '.zip' else Path(str(path) + '.zip')


class RLOffsetController:
    def __init__(self, config):
        path = offset_model_path_from_config(config)
        if not path.is_file():
            raise FileNotFoundError(f'RL offset model not found: {path}. '
                                    'Run train_rl.py with config/rl_training_offset.yaml first.')
        metadata_path = path.with_suffix('.json')
        if not metadata_path.is_file():
            raise FileNotFoundError(f'RL offset model metadata not found: {metadata_path}')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        self.spec = OffsetSpec.from_config(config)
        from simulation.engine import build_simulation_params  # lazy: engine imports this module
        self.params = build_simulation_params(config)
        expected = self.spec.metadata()
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError('RL offset model observation/action contract differs from configuration; retrain the model')
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
        if (self.model.observation_space.shape != (self.spec.observation_size,)
                or self.model.action_space.n != len(self.spec.shifts_K)):
            raise ValueError('RL offset model has incompatible observation/action spaces')

    def shift_K(self, state, inputs):
        import numpy as np
        values = offset_observation(state, inputs.outside_temperature_C, inputs.timestamp,
                                    self.params, self.spec)
        action, _ = self.model.predict(np.asarray(values, dtype=np.float32), deterministic=True)
        return self.spec.shifts_K[int(action)]


class ConstantShiftController:
    """Reference for the offset agent: the same set-point shift at every decision, no model needed.

    Select it with control.reinforcement_learning_offset.constant_shift_K (0 = plain heating_curve)."""

    def __init__(self, value_K):
        if not isfinite(value_K):
            raise ValueError('control.reinforcement_learning_offset.constant_shift_K must be finite')
        self.value_K = float(value_K)

    def shift_K(self, state, inputs):
        return self.value_K
