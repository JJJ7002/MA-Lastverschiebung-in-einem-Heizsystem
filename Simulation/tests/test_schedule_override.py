from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support import load_test_config
from control.temperature_targets import HeatingCurveParams
from control.operating_constraints import OperatingParams, OperatingState, ScheduleOverride
from weather.loader import WeatherPoint
from simulation.engine import run_simulation




class ScheduleTests(unittest.TestCase):
    def schedule(self, mode='force_off', start='22:00', end='06:00'):
        return ScheduleOverride.from_config({'enabled': True, 'windows': [
            {'start': start, 'end': end, 'mode': mode}
        ]})

    def update(self, schedule, hour, temperature, state, **kwargs):
        params = OperatingParams(HeatingCurveParams(-0.5, 32),
                                 schedule=schedule)
        from control.rl_power import apply_power_request
        request = .5 if temperature < 42 else -1
        return apply_power_request(request, temperature, state, params,
            replace(self.heat_pump(), **{'min_on_time_min': 0, 'min_off_time_min': 0, **kwargs}), 10,
            datetime(2020, 1, 1, hour))[0]

    @staticmethod
    def heat_pump():
        # The command is quantized to the manufacturer map, so a real map is required.
        from simulation.engine import build_simulation_params
        return build_simulation_params(load_test_config()).heat_pump


    def test_boundaries_and_midnight(self):
        for start, end, expected in [('22:00', '06:00', [True, True, False, False, True]),
                                     ('06:00', '22:00', [False, False, True, True, False])]:
            schedule = self.schedule(start=start, end=end)
            actual = [schedule.mode_at(datetime(2020, 1, 1, hour)) is not None
                      for hour in (0, 5, 6, 21, 22)]
            self.assertEqual(actual, expected)

    def test_blocking_overrides_minimum_on_time_and_retains_off_timer(self):
        result = self.update(self.schedule(), 23, 20, OperatingState(True, 0), min_on_time_min=60)
        self.assertEqual(result, OperatingState(False, 0))
        result = self.update(self.schedule(), 23, 20, result)
        self.assertEqual(result, OperatingState(False, 10))

    def test_forced_on_ignores_temperature_but_respects_off_time(self):
        schedule = self.schedule('force_on')
        result = self.update(schedule, 23, 50, OperatingState(False, 0), min_off_time_min=30)
        self.assertFalse(result.is_on)
        result = self.update(schedule, 23, 50, OperatingState(False, 20), min_off_time_min=30)
        self.assertEqual(result, OperatingState(True, 0))
        self.assertTrue(self.update(schedule, 23, 50, result).is_on)
        self.assertFalse(self.update(schedule, 6, 50, result).is_on)

    def test_blocking_wins_over_overlap(self):
        on = self.schedule('force_on')
        off = self.schedule()
        for windows in (on.windows + off.windows, off.windows + on.windows):
            schedule = ScheduleOverride(True, windows)
            self.assertFalse(self.update(schedule, 23, 20, OperatingState(True, 100)).is_on)

    def test_disabled_and_outside_window_use_hysteresis(self):
        schedule = self.schedule()
        self.assertTrue(self.update(schedule, 12, 20, OperatingState()).is_on)
        self.assertTrue(self.update(replace(schedule, enabled=False), 23, 20, OperatingState()).is_on)
        self.assertIsNone(replace(schedule, enabled=False).mode_at(None))
        with self.assertRaises(ValueError):
            schedule.mode_at(None)

    def test_invalid_windows(self):
        for args in ({'mode': 'off'}, {'start': '25:00'}, {'start': '06:00'}, {'end': 'bad'}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.schedule(**args)

    def test_seasonal_heating_off_repeats_each_year_and_forces_off(self):
        schedule = ScheduleOverride.from_config({
            'enabled': False,
            'seasonal_heating_off': {'enabled': True, 'start': '06-01', 'end': '09-01'},
        })
        for year in (2024, 2025, 2026):
            for month in (6, 7, 8):
                instant = datetime(year, month, 15, 12)
                self.assertTrue(schedule.heating_off_at(instant))
                self.assertEqual(schedule.mode_at(instant), 'force_off')
            self.assertFalse(schedule.heating_off_at(datetime(year, 5, 31, 23, 59)))
            self.assertFalse(schedule.heating_off_at(datetime(year, 9, 1, 0, 0)))

    def test_seasonal_shutdown_disables_heat_pump_and_heating_circuit(self):
        from simulation.engine import build_simulation_params, initial_state
        from simulation.step import SimulationInput, simulate_step

        config = load_test_config()
        config['simulation'].update(settling_hours=0, start_time=None, end_time=None)
        config['control']['method'] = 'heating_curve_supply'
        config['control']['schedule'] = {'seasonal_heating_off': {'enabled': True}}
        params = build_simulation_params(config)
        state = initial_state(config, 0)
        state.heat_pump = OperatingState(True, 60)
        result = simulate_step(state, SimulationInput(0, datetime(2025, 6, 1)), params)

        self.assertFalse(result.state.heat_pump.is_on)
        self.assertEqual(result.thermal_power_W, 0)
        self.assertEqual(result.electric_power_W, 0)
        self.assertEqual(result.heating_power_to_building_W, 0)

    def test_engine_passes_simulation_time_and_blocked_pump_uses_no_energy(self):
        config = load_test_config()
        config['weather']['source'] = 'csv'  # Isolate model tests from network access
        config['simulation']['weather_file'] = 'data/weather/ogd-smn_bas_t_historical_2020-2029.csv'
        config['simulation'].update(start_time=None, end_time=None)
        config['simulation']['dt_minutes'] = 10
        config['control']['method'] = 'heating_curve'
        config['storage']['T_storage_init_C'] = 35
        config['control']['schedule'] = {'enabled': True, 'windows': [
            {'start': '11:00', 'end': '11:20', 'mode': 'force_off'}
        ]}
        config['control']['heating_curve']['offset_K'] = 100
        start = datetime(2020, 1, 1, 10, 50)
        with patch('simulation.engine.load_weather_csv', return_value=[
            WeatherPoint(start + timedelta(minutes=10 * index), 0) for index in range(4)
        ]):
            result = run_simulation(config)
        self.assertEqual(result.heat_pump_on, [True, False, False, True])
        self.assertEqual(result.electric_power_W[1:3], [0, 0])


def test_engine_skips_seasonal_pause_and_restarts():
    from datetime import timedelta
    from weather.loader import WeatherPoint
    from simulation.engine import run_simulation

    config = load_test_config()
    config['simulation'].update(settling_hours=0, start_time=None, end_time=None, dt_minutes=10)
    config['control']['method'] = 'heating_curve'
    config['control']['schedule'] = {'seasonal_heating_off': {'enabled': True, 'start': '06-01', 'end': '09-01'}}
    pts = [WeatherPoint(datetime(2024, 5, 31) + timedelta(minutes=10 * i), 2) for i in range(144 * 2)]
    pts += [WeatherPoint(datetime(2024, 9, 1) + timedelta(minutes=10 * i), 2) for i in range(144 * 2)]
    result = run_simulation(config, weather_points=pts)
    # 31 May (144) heated, 1 June skipped, 1-2 Sep (288) heated
    assert len(result.heat_pump_on) == 144 + 288
    assert len(result.segment_starts) == 2
