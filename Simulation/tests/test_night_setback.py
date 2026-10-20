from datetime import datetime
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from support import load_test_config
from control.temperature_targets import NightSetback
from weather.loader import WeatherPoint
from simulation.engine import run_simulation
from simulation.engine import build_result_rows
from analysis.metrics import calculate_metrics
from analysis.scoring import ScoringParams, calculate_score


class NightSetbackTests(unittest.TestCase):
    def test_warmup_longer_than_night_is_not_clamped(self):
        from datetime import timedelta
        start = datetime(2020, 1, 1, 22)
        for rounding in (0, .5, 1):
            for total_minutes in (960, 1920):
                setback = NightSetback(enabled=True, room_offset_K=2,
                    warmup_minutes_pro_K=total_minutes / 2, warmup_rounding=rounding)
                for elapsed in (0, 60, 240, 479):
                    instant = start + timedelta(minutes=elapsed)
                    x = (480 - elapsed) / total_minutes
                    expected = 2 * ((1 - rounding) * x
                                    + rounding * x**3 * (10 - 15*x + 6*x*x))
                    self.assertAlmostEqual(setback.room_reduction_K(instant), expected)
                    self.assertEqual(setback.room_reduction_K(instant, comfort=True), 2)
                self.assertLess(setback.room_reduction_K(start), 2)
                self.assertEqual(setback.room_reduction_K(start - timedelta(minutes=1)), 0)
                self.assertEqual(setback.room_reduction_K(start + timedelta(hours=8)), 0)
                self.assertEqual(setback.room_reduction_K(start + timedelta(hours=8), comfort=True), 0)

    def test_warmup_duration_scales_with_room_setback(self):
        for offset, duration in [(0, 0), (1, 300), (2, 600), (3, 900)]:
            setback = NightSetback.from_config({'room_offset_K': offset,
                                                'warmup_minutes_pro_K': 300})
            self.assertEqual(setback.warmup_minutes, duration)
        with self.assertRaisesRegex(ValueError, 'warmup_minutes_pro_K'):
            NightSetback.from_config({'warmup_minutes': 600})

    def test_room_and_comfort_targets_at_boundaries(self):
        setback = NightSetback(enabled=True)
        for hour, minute, room, comfort in [(21, 59, 0, 0), (22, 0, 2, 2),
                (3, 59, 2, 2), (4, 0, 2, 2), (5, 0, 1, 2),
                (5, 59, 2 * (1/120)**3 * (10 - 15/120 + 6/120**2), 2),
                (6, 0, 0, 0)]:
            timestamp = datetime(2020, 1, 1, hour, minute)
            self.assertEqual(setback.room_reduction_K(timestamp), room)
            self.assertEqual(setback.room_reduction_K(timestamp, comfort=True), comfort)
        late = NightSetback.from_config({'enabled': True, 'start': '18:00', 'end': '23:00'})
        self.assertEqual(late.room_reduction_K(datetime(2020, 1, 1, 22), comfort=True), 2)
        self.assertEqual(late.room_reduction_K(datetime(2020, 1, 1, 22)), 1)
        self.assertEqual(late.room_reduction_K(datetime(2020, 1, 2, 0), comfort=True), 0)
        self.assertEqual(late.room_reduction_K(datetime(2020, 1, 2, 1), comfort=True), 0)
        immediate = NightSetback(enabled=True, warmup_minutes_pro_K=0)
        self.assertEqual(immediate.room_reduction_K(datetime(2020, 1, 1, 6), comfort=True), 0)

    def test_room_demand_comfort_cost_and_csv(self):
        config = load_test_config()
        config['weather']['source'] = 'csv'  # Isolate model tests from network access
        config['simulation']['weather_file'] = 'data/weather/ogd-smn_bas_t_historical_2020-2029.csv'
        config['simulation'].update(start_time=None, end_time=None)
        config['simulation']['dt_minutes'] = 60
        config['simulation']['output_interval_minutes'] = 60
        config['control']['interval_minutes'] = 60
        config['control']['method'] = 'heating_curve'
        config['control']['night_setback'] = {'enabled': True, 'start': '22:00',
            'end': '06:00', 'offset_K': 3, 'room_offset_K': 2, 'warmup_minutes_pro_K': 60}
        config['control']['schedule']['enabled'] = False
        config['building']['T_set_C'] = 21
        config['building']['T_room_init_C'] = 19.5
        config['building']['comfort_buffer_K'] = 1
        config['storage']['T_storage_init_C'] = 50
        with patch('simulation.engine.load_weather_csv', return_value=[
            WeatherPoint(datetime(2020, 1, 1, hour), 0) for hour in (3, 5, 6)
        ]):
            result = run_simulation(config)
        self.assertLess(result.room_temperatures_C[0], 20)
        self.assertGreater(result.room_temperatures_C[1], result.room_temperatures_C[0])
        rows = build_result_rows(result)
        self.assertEqual([row['target_room_temperature_C'] for row in rows], [19, 20, 21])
        self.assertEqual([row['comfort_threshold_C'] for row in rows], [18, 18, 20])
        result.room_temperatures_C[:] = [18, 18.5, 19]
        metrics = calculate_metrics(result, 1, .25, 21)
        self.assertEqual(metrics.comfort_below_hours, .25)
        scoring = ScoringParams.from_config(config['scoring'])
        self.assertEqual(calculate_score(result, metrics, .25, scoring).comfort_cost,
                         .25 * scoring.cost_per_comfort_degree_hour)

    def test_warmup_before_midnight_and_full_night_warmup(self):
        setback = NightSetback.from_config({'enabled': True, 'start': '20:00',
                                            'end': '01:00', 'warmup_minutes_pro_K': 60})
        for hour, room, comfort in [(22, 2, 2), (23, 2, 2), (0, 1, 2), (1, 0, 0)]:
            instant = datetime(2020, 1, 1, hour)
            self.assertEqual(setback.room_reduction_K(instant), room)
            self.assertEqual(setback.room_reduction_K(instant, comfort=True), comfort)
        full = NightSetback(enabled=True, warmup_minutes_pro_K=240)
        instant = datetime(2020, 1, 1, 23)
        self.assertEqual(full.room_reduction_K(instant), 1.9678955078125)
        self.assertEqual(full.reduction_K(instant), 0)
        self.assertEqual(full.room_reduction_K(instant, comfort=True), 2)

    def test_smooth_warmup_is_monotone_with_gentle_endpoints(self):
        from datetime import timedelta
        setback = NightSetback(enabled=True)
        start = datetime(2020, 1, 1, 4)
        targets = [21 - setback.room_reduction_K(start + timedelta(minutes=i))
                   for i in range(121)]
        self.assertEqual(targets[0], 19)
        self.assertEqual(targets[-1], 21)
        self.assertAlmostEqual(targets[30], 19.20703125)
        self.assertAlmostEqual(targets[90], 20.79296875)
        # Stronger rounding than cubic smoothstep at a quarter of the ramp.
        self.assertLess(targets[30], 19.3125)
        self.assertGreater(targets[90], 20.6875)
        increments = [b - a for a, b in zip(targets, targets[1:])]
        self.assertTrue(all(delta > 0 for delta in increments))
        self.assertLess(increments[0], (2 / 120) / 10)
        self.assertAlmostEqual(increments[0], increments[-1])
        self.assertGreater(increments[60], increments[0])

    def test_configurable_warmup_rounding(self):
        from datetime import timedelta
        start = datetime(2020, 1, 1, 4)
        for rounding in (0, 0.25, 0.5, 1):
            setback = NightSetback.from_config({'enabled': True,
                                                'warmup_rounding': rounding})
            targets = [21 - setback.room_reduction_K(start + timedelta(minutes=i))
                       for i in range(121)]
            with self.subTest(rounding=rounding):
                self.assertEqual(targets[0], 19)
                self.assertEqual(targets[-1], 21)
                self.assertAlmostEqual(targets[30], 19.5 * (1-rounding) + 19.20703125 * rounding)
                self.assertAlmostEqual(targets[90], 20.5 * (1-rounding) + 20.79296875 * rounding)
                self.assertTrue(all(b > a for a, b in zip(targets, targets[1:])))
                self.assertEqual(setback.room_reduction_K(start + timedelta(minutes=30), comfort=True), 2)
        for value in (-0.1, 1.1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'warmup_rounding'):
                NightSetback.from_config({'warmup_rounding': value})

    def test_window_boundaries_and_midnight(self):
        setback = NightSetback.from_config({'enabled': True})
        for hour, minute, expected in [(21, 59, 0), (22, 0, 3), (23, 59, 3),
                                       (0, 0, 3), (3, 59, 3), (4, 0, 0), (6, 0, 0)]:
            with self.subTest(hour=hour, minute=minute):
                self.assertEqual(setback.reduction_K(datetime(2020, 1, 1, hour, minute)), expected)
        daytime = NightSetback.from_config({'enabled': True, 'start': '10:00', 'end': '12:00', 'warmup_minutes_pro_K': 0})
        for hour, expected in [(9, 0), (10, 3), (11, 3), (12, 0)]:
            self.assertEqual(daytime.reduction_K(datetime(2020, 1, 1, hour)), expected)
        self.assertEqual(NightSetback().reduction_K(None), 0)
        with self.assertRaises(ValueError):
            setback.reduction_K(None)


    def test_config_reaches_simulation(self):
        config = load_test_config()
        config['weather']['source'] = 'csv'  # Isolate model tests from network access
        config['simulation']['weather_file'] = 'data/weather/ogd-smn_bas_t_historical_2020-2029.csv'
        config['simulation'].update(start_time=None, end_time=None)
        config['control']['method'] = 'heating_curve'
        config['control']['schedule']['enabled'] = False
        config['control']['heating_curve'] = {'slope_factor': 1, 'offset_K': -6, 'temperature_gain_W_per_K': 5000}
        config['heat_pump']['min_off_time_min'] = 0
        config['control']['night_setback'] = {'start': '22:00', 'end': '06:00', 'room_offset_K': 2, 'warmup_minutes_pro_K': 0}
        config['storage']['T_storage_init_C'] = 39
        with patch('simulation.engine.load_weather_csv', return_value=[
            WeatherPoint(datetime(2020, 1, 1, 23), 0)
        ]):
            config['control']['night_setback']['enabled'] = True
            night = run_simulation(config)
            config['control']['night_setback']['enabled'] = False
            normal = run_simulation(config)
        self.assertEqual(night.thermal_power_W, [0])
        self.assertGreater(normal.thermal_power_W[0], 0)

    def test_invalid_config(self):
        for config in [{'enabled': 'true'}, {'start': '06:00', 'end': '06:00'},
                       {'start': '24:00'}, {'end': 6}, {'start': '6:00'},
                       {'offset_K': -1}, {'offset_K': float('nan')}, {'offset_K': float('inf')}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                NightSetback.from_config(config)
        for key in ('room_offset_K', 'warmup_minutes_pro_K'):
            for value in (-1, float('nan'), float('inf')):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    NightSetback.from_config({key: value})
