from dataclasses import replace
from datetime import datetime, time, timedelta
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support import load_test_config
from control.rl_power import (apply_power_request, observation,
                                            power_levels_from_config, RLController, OBSERVATION_SIZE)
from control.operating_constraints import OperatingState, ScheduleOverride, ScheduleWindow
from weather.loader import WeatherPoint
from simulation.engine import build_simulation_params, initial_state, run_simulation
from simulation.step import SimulationInput


class RLControlTests(unittest.TestCase):
    def setUp(self):
        self.config = load_test_config()
        # Use the fixed two-node building model for these controller regression assertions.
        self.config['weather']['source'] = 'csv'  # Isolate model tests from network access
        self.config['simulation']['weather_file'] = 'data/weather/ogd-smn_bas_t_historical_2020-2029.csv'
        self.config['simulation']['dt_minutes'] = 10
        self.config['simulation'].update(start_time=None, end_time=None, settling_hours=0)
        self.config['heat_pump'].update(min_on_time_min=0, min_off_time_min=0)
        self.params = build_simulation_params(self.config)

    def observe(self, state, timestamp, outside=0.0):
        return observation(state, outside, timestamp, self.params)

    def test_observation_time_wrap(self):
        state = initial_state(self.config, 0)
        a = self.observe(state, datetime(2020, 1, 1, 23, 59))
        b = self.observe(state, datetime(2020, 1, 2))
        self.assertEqual(len(a), OBSERVATION_SIZE)
        self.assertLess(abs(a[2] - b[2]), .005)
        self.assertLess(abs(a[3] - b[3]), .005)

    def test_observation_uses_storage_profile_and_supply_margin(self):
        self.config['storage']['initial_node_temperatures_C'] = [60] + [50] * 8 + [30]
        state = initial_state(self.config, 0)
        values = self.observe(state, datetime(2025, 1, 1, 12))
        self.assertAlmostEqual(values[4], (60 - 45) / 20)
        self.assertAlmostEqual(values[5], (50 - 40) / 20)
        self.assertAlmostEqual(values[6], (30 - 35) / 20)
        self.assertGreater(values[7], 0)
        cold = initial_state({**self.config, 'storage': {**self.config['storage'],
                              'initial_node_temperatures_C': [25] * 10}}, 0)
        self.assertLess(self.observe(cold, datetime(2025, 1, 1, 12))[7], 0)

    def test_inference_uses_storage_profile_observation(self):
        self.config['storage']['initial_node_temperatures_C'] = [60] + [30] * 9
        state = initial_state(self.config, 0)
        controller = RLController.__new__(RLController)
        controller.power_levels = (-1, 0, .4, .6, .8, 1)
        controller.params = self.params
        from unittest.mock import Mock
        controller.model = Mock()
        controller.model.predict.return_value = (0, None)
        controller.predict(state, SimulationInput(0, datetime(2025, 1, 1, 12)))
        values = controller.model.predict.call_args.args[0]
        self.assertEqual(len(values), OBSERVATION_SIZE)
        self.assertAlmostEqual(values[4], (60 - 45) / 20)
        self.assertAlmostEqual(values[6], (30 - 35) / 20)

    def test_target_deviation_and_target_follow_setback(self):
        state = initial_state(self.config, 0)  # room 21 C
        day = self.observe(state, datetime(2020, 1, 1, 12))
        self.assertAlmostEqual(day[1], 0)
        self.assertAlmostEqual(day[8], self.params.target_room_temperature_C - 20)
        night_time = datetime(2020, 1, 1, 23)
        night = self.observe(state, night_time)
        # current reduction of the room target, including a warm-up ramp that may already have started
        offset = self.params.operating.night_setback.room_reduction_K(night_time)
        self.assertGreater(offset, 0)
        self.assertAlmostEqual(night[1], offset / .5)  # room is `offset` above the lowered target
        self.assertAlmostEqual(night[8], self.params.target_room_temperature_C - offset - 20)

    def test_constraints_minimum_times_power_and_ceiling(self):
        control = self.params.operating
        hp = replace(self.params.heat_pump, min_on_time_min=30, min_off_time_min=30)
        def apply(power, state, temp=40):
            return apply_power_request(power, temp, state, control, hp, 10, datetime(2020, 1, 1))
        self.assertEqual(apply(1, OperatingState(False, 10))[1], -1)
        self.assertEqual(apply(.1, OperatingState(False, 20))[1], .1)
        self.assertEqual(apply(-1, OperatingState(True, 10))[1], 0)
        self.assertEqual(apply(-1, OperatingState(True, 20))[1], -1)
        self.assertEqual(apply(0, OperatingState(False, 20))[1], 0)
        self.assertEqual(apply(1, OperatingState(True, 0), 65)[1], -1)

    def test_schedule_priority(self):
        for mode, expected in [('force_off', -1), ('force_on', 1)]:
            control = replace(self.params.operating, schedule=ScheduleOverride(
                True, (ScheduleWindow(time(0), time(1), mode),)))
            state, power = apply_power_request(.6, 40, OperatingState(), control, replace(self.params.heat_pump, min_off_time_min=0), 10,
                                                datetime(2020, 1, 1))
            self.assertEqual(power, expected)
            self.assertEqual(state.is_on, expected >= 0)
            self.assertEqual(apply_power_request(1, 65, state, control, replace(self.params.heat_pump, min_off_time_min=0), 10,
                                                 datetime(2020, 1, 1))[1], -1)

    def test_reject_invalid_action_levels(self):
        for levels in ([0, .2, 1], [0, .6, .4, 1], [0, 1.2], [0, float('nan'), 1]):
            self.config['control']['reinforcement_learning']['power_levels'] = levels
            with self.assertRaises(ValueError):
                power_levels_from_config(self.config)

    def test_missing_model_explains_training(self):
        self.config['control']['reinforcement_learning']['model_path'] = 'missing-test-model.zip'
        with self.assertRaisesRegex(FileNotFoundError, 'train_rl.py'):
            RLController(self.config)

    def test_engine_dispatch_and_observation_of_executed_power(self):
        self.config['control']['method'] = 'reinforcement_learning'
        seen = []
        def predict(state, inputs):
            seen.append(state.power_fraction)
            return .6 if len(seen) == 1 else -1
        weather = [WeatherPoint(datetime(2020, 1, 1) + timedelta(minutes=10*i), 0) for i in range(2)]
        with patch('control.rl_power.RLController') as controller:
            controller.return_value.predict.side_effect = predict
            result = run_simulation(self.config, weather_points=weather)
        self.assertAlmostEqual(result.heat_pump_power_fractions[0], .6)
        self.assertEqual(result.heat_pump_power_fractions[1], -1)
        self.assertAlmostEqual(seen[1], .6)
        self.assertEqual(seen[0], -1)
        self.assertGreater(result.electric_power_W[0], 0)
        self.assertEqual(result.electric_power_W[1], 0)

    def test_old_controller_does_not_load_rl_model(self):
        self.config['control']['method'] = 'heating_curve'
        with patch('control.rl_power.RLController', side_effect=AssertionError('unexpected load')):
            result = run_simulation(self.config, weather_points=[WeatherPoint(datetime(2020, 1, 1), 0)])
        self.assertEqual(len(result.timestamps), 1)


def test_observation_contains_previous_power():
    from dataclasses import replace
    from datetime import datetime
    from simulation.engine import build_simulation_params, initial_state
    c = load_test_config()
    p = build_simulation_params(c)
    off = initial_state(c, 0)
    on = replace(off, power_fraction=0.4)
    now = datetime(2025, 1, 1, 12)
    a = observation(off, 0, now, p)
    b = observation(on, 0, now, p)
    assert a[-1] == -1 and b[-1] == 0.4 and a[:-1] == b[:-1]
