from dataclasses import replace
from pathlib import Path
import sys
import unittest
from datetime import datetime
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from control.cop_optimized import CopOptimizedParams, power_from_advantage, update_cop_control
from control.operating_constraints import OperatingParams, OperatingState, ScheduleOverride
from control.temperature_targets import HeatingCurveParams
from support import load_test_config
from simulation.engine import run_simulation, _build_heat_pump_params
from weather.loader import WeatherPoint, load_daily_temperatures
from models.heat_pump import get_performance


class CopControlTests(unittest.TestCase):
    def test_fallback_reporting_distinguishes_active_and_override(self):
        from analysis.metrics import calculate_metrics
        from simulation.engine import build_result_rows
        config = load_test_config()
        config['weather']['source'] = 'csv'
        config['simulation']['weather_file'] = 'data/weather/ogd-smn_bas_t_historical_2020-2029.csv'
        config['control']['method'] = 'cop_optimized'
        config['simulation'].update(start_time=None, end_time=None)
        config['control']['night_setback']['enabled'] = False
        config['control']['schedule']['enabled'] = False
        config['control']['heating_curve'] = {'slope': 0, 'offset_C': 40}
        config['heat_pump']['min_off_time_min'] = 0
        config['storage']['T_storage_init_C'] = 35
        timestamp = datetime(2020, 1, 1, 12)
        with patch('simulation.engine.load_weather_csv', return_value=[WeatherPoint(timestamp, 0)]), patch(
                'simulation.engine.load_daily_temperatures', return_value={timestamp.date(): 0}):
            config['control']['cop_optimized'].update(min_power_advantage=10, full_power_advantage=20)
            result = run_simulation(config)
            metrics = calculate_metrics(result, 1, .25, 21)
            self.assertEqual(metrics.fallback_active_hours, .25)
            self.assertEqual(metrics.fallback_override_hours, .25)
            self.assertTrue(build_result_rows(result)[0]['fallback_override'])
            config['control']['cop_optimized'].update(min_power_advantage=-.9, full_power_advantage=-.8)
            result = run_simulation(config)
            metrics = calculate_metrics(result, 1, .25, 21)
            self.assertEqual(metrics.fallback_active_hours, .25)
            self.assertEqual(metrics.fallback_override_hours, 0)
            config['control']['schedule'] = {'enabled': True, 'windows': [
                {'start': '11:00', 'end': '13:00', 'mode': 'force_on'}]}
            result = run_simulation(config)
            self.assertEqual(result.fallback_active, [False])
            self.assertEqual(result.fallback_override, [False])
            config['control']['method'] = 'heating_curve'
            result = run_simulation(config)
            self.assertEqual(calculate_metrics(result, 1, .25, 21).fallback_active_hours, 0)

    def test_heating_curve_fallback_uses_shared_request(self):
        from simulation.engine import build_simulation_params, initial_state
        from simulation.step import simulate_step, SimulationInput
        from control.heating_curve import heating_curve_power_request, heating_supply_curve_power_request
        from models.storage import StorageState
        requests = {'heating_curve_supply': heating_supply_curve_power_request,
                    'heating_curve': heating_curve_power_request}
        for fallback, fallback_request in requests.items():
            config = load_test_config()
            config['control']['method'] = 'cop_optimized'
            config['control']['cop_optimized']['fallback'] = fallback
            config['control']['night_setback']['enabled'] = False
            config['control']['schedule']['enabled'] = False
            config['heat_pump'].update(min_on_time_min=0, min_off_time_min=0)
            params = build_simulation_params(config)
            instant = datetime(2020, 1, 1, 12)
            inputs = SimulationInput(0, instant)
            params.cop_control = CopOptimizedParams({instant.date(): 100})
            for storage, room in [(35, 20), (50, 21), (64, 22)]:
                with self.subTest(fallback=fallback, storage=storage):
                    state = initial_state(config, inputs.outside_temperature_C)
                    state.storage = StorageState(storage)
                    state.building.T_room_C = room
                    state.power_fraction = .4
                    state.heat_pump = OperatingState(True, 60)
                    cop = simulate_step(state, inputs, params)
                    expected = fallback_request(state, inputs, params)[0].power_fraction
                    self.assertAlmostEqual(cop.heat_pump_commanded_power_fraction, expected)
            # COP demand still wins when larger than the fallback.
            params.cop_control = CopOptimizedParams({instant.date(): 1})
            state.storage = StorageState(35)
            result = simulate_step(state, inputs, params)
            self.assertEqual(result.power_fraction, 1)
            self.assertTrue(result.state.heat_pump.fallback_running)
            self.assertFalse(result.state.heat_pump.fallback_override)

    def test_heating_curve_fallback_obeys_minimum_power_timers_and_safety(self):
        config = load_test_config()
        hp = replace(_build_heat_pump_params(config), min_on_time_min=30, min_off_time_min=30)
        instant = datetime(2020, 1, 1, 12)
        cop = CopOptimizedParams({instant.date(): 100})
        control = OperatingParams(HeatingCurveParams(0, 40))
        def update(state, request, storage=40, settings=control):
            return update_cop_control(storage, 0, instant, state, settings, hp, cop,
                                      10, fallback_power=request)
        self.assertEqual(update(OperatingState(False, 0), .1)[1], -1)
        self.assertEqual(update(OperatingState(False, 20), .1)[1], .1)
        self.assertEqual(update(OperatingState(True, 0), -1)[1], 0)
        self.assertEqual(update(OperatingState(True, 30), -1)[1], -1)
        self.assertEqual(update(OperatingState(True, 30), 0)[1], 0)
        self.assertEqual(update(OperatingState(True, 0), 1, hp.sink_temp_max_C)[1], -1)
        blocked = replace(control, schedule=ScheduleOverride.from_config({
            'enabled': True, 'windows': [{'start': '11:00', 'end': '13:00', 'mode': 'force_off'}]}))
        self.assertEqual(update(OperatingState(True, 0), 1, settings=blocked)[1], -1)

    def test_linear_mapping(self):
        for advantage, expected in [(-.01, -1), (0, 0), (.1, .5), (.2, 1), (.4, 1)]:
            self.assertAlmostEqual(power_from_advantage(advantage, .2), expected)

    def test_adjustable_minimum_threshold(self):
        for lower, upper in [(.05, .15), (-.1, .1), (-.2, -.1)]:
            CopOptimizedParams({}, upper, lower)
            self.assertEqual(power_from_advantage(lower - .001, upper, lower), -1)
            self.assertEqual(power_from_advantage(lower, upper, lower), 0)
            self.assertAlmostEqual(power_from_advantage((lower + upper) / 2, upper, lower), .5)
            self.assertEqual(power_from_advantage(upper, upper, lower), 1)
        for lower, upper in [(0, 0), (.2, .1), (-1, .2), (float('nan'), .2), (0, float('inf'))]:
            with self.assertRaises(ValueError):
                CopOptimizedParams({}, upper, lower)

    def test_feedback_timers_schedule_and_ceiling(self):
        config = load_test_config()
        config['weather']['source'] = 'csv'
        config['simulation']['weather_file'] = 'data/weather/ogd-smn_bas_t_historical_2020-2029.csv'
        hp = replace(_build_heat_pump_params(config), min_on_time_min=0, min_off_time_min=0)
        timestamp = datetime(2020, 1, 1, 12)
        reference = get_performance(0, 40, hp)[0]
        cop = CopOptimizedParams({timestamp.date(): reference})
        hp = replace(hp, min_on_time_min=30, min_off_time_min=30)
        control = OperatingParams(HeatingCurveParams(-.7, 30))
        def update(temp, state, params=control):
            return update_cop_control(temp, 0, timestamp, state, params, hp, cop, 10)
        self.assertEqual(update(40, OperatingState(False, 0))[1], -1)
        self.assertAlmostEqual(update(40, OperatingState(False, 20))[1], 0)
        self.assertGreater(update(35, OperatingState(True, 30))[1], 0)
        self.assertEqual(update(45, OperatingState(True, 30))[1], -1)
        self.assertEqual(update(45, OperatingState(True, 0))[1], 0)
        self.assertEqual(update(hp.sink_temp_max_C, OperatingState(True, 0))[1], -1)
        for mode, expected in [('force_off', -1), ('force_on', 1)]:
            scheduled = OperatingParams(HeatingCurveParams(-.7, 40),
                schedule=ScheduleOverride.from_config({'enabled': True, 'windows': [
                    {'start': '11:00', 'end': '13:00', 'mode': mode}]}))
            self.assertEqual(update(45, OperatingState(True, 30), scheduled)[1], expected)

    def test_daily_reference_changes_only_at_date_boundary_and_missing_day_fails(self):
        config = load_test_config()
        config['weather']['source'] = 'csv'
        config['simulation']['weather_file'] = 'data/weather/ogd-smn_bas_t_historical_2020-2029.csv'
        config['simulation']['dt_minutes'] = 60
        config['simulation']['output_interval_minutes'] = 60
        config['control']['interval_minutes'] = 60
        config['control']['method'] = 'cop_optimized'
        config['simulation'].update(start_time=None, end_time=None)
        weather = [WeatherPoint(datetime(2020, 1, 1, 12), 0),
                   WeatherPoint(datetime(2020, 1, 1, 13), 5),
                   WeatherPoint(datetime(2020, 1, 2, 0), 0)]
        daily = {weather[0].timestamp.date(): 0, weather[2].timestamp.date(): 5}
        with patch('simulation.engine.load_weather_csv', return_value=weather), patch(
                'simulation.engine.load_daily_temperatures', return_value=daily):
            result = run_simulation(config)
            self.assertEqual(result.reference_cop[0], result.reference_cop[1])
            self.assertNotEqual(result.reference_cop[1], result.reference_cop[2])
            del daily[weather[2].timestamp.date()]
            with self.assertRaisesRegex(ValueError, 'Missing tre200d0'):
                run_simulation(config)

    def test_actual_daily_file(self):
        daily = load_daily_temperatures(Path(__file__).resolve().parents[1] / 'data/weather/ogd-smn_bas_d_historical.csv')
        self.assertIn(datetime(2020, 1, 1).date(), daily)
