from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.metrics import calculate_metrics
from analysis.scoring import ScoringParams, calculate_score
from simulation.engine import SimulationResult


class ScoringTests(unittest.TestCase):
    def test_starts_per_day_uses_simulated_duration(self):
        result = SimulationResult(
            timestamps=[], outside_temperatures_C=[], room_temperatures_C=[21] * 4,
            storage_temperatures_C=[], heat_pump_on=[True, True, False, True],
            cop=[], thermal_power_W=[], electric_power_W=[],
            total_electric_energy_Wh=0, total_thermal_energy_Wh=0,
        )
        for dt, expected in [(1, 12), (.5, 24), (12, 1)]:
            metrics = calculate_metrics(result, 1, dt, 21)
            self.assertEqual(metrics.heat_pump_start_count, 2)
            self.assertEqual(metrics.heat_pump_starts_per_day, expected)
        result.heat_pump_on.clear()
        result.room_temperatures_C.clear()
        self.assertEqual(calculate_metrics(result, 1, 1, 21).heat_pump_starts_per_day, 0)

    def comfort_result(self, rooms):
        n = len(rooms)
        return SimulationResult(
            timestamps=[], outside_temperatures_C=[0] * n, room_temperatures_C=rooms,
            storage_temperatures_C=[35] * n, heat_pump_on=[False] * n,
            cop=[], thermal_power_W=[], electric_power_W=[],
            total_electric_energy_Wh=0, total_thermal_energy_Wh=0,
        )

    def test_comfort_duration_uses_buffer(self):
        result = self.comfort_result([21, 20.5, 19.5, 21.5])
        self.assertEqual(calculate_metrics(result, 1, 0.25, 21).comfort_below_hours, 0.25)

    def test_comfort_buffer_boundary_and_changed_target(self):
        result = self.comfort_result([19.9, 20.0, 20.1, 21.0])
        metrics = calculate_metrics(result, 1, 0.25, 21)
        self.assertEqual(metrics.comfort_below_hours, 0.25)
        self.assertEqual(metrics.comfort_threshold_C, 20)
        self.assertEqual(metrics.comfort_buffer_K, 1)
        self.assertEqual(calculate_metrics(result, 1, 0.25, 22).comfort_below_hours, 0.75)
        self.assertEqual(calculate_metrics(result, 0, 0.25, 21).comfort_below_hours, 0.75)
        for buffer in (-1, float('nan'), float('inf')):
            with self.subTest(buffer=buffer), self.assertRaises(ValueError):
                calculate_metrics(result, buffer, 0.25, 21)

    def setUp(self):
        self.config = {
            'currency': 'CHF', 'cost_per_heat_pump_start': 0.05,
            'cost_per_comfort_degree_hour': 3,
            'electricity': {'default_price_per_kWh': 0.3, 'tariff_windows': [
                {'start': '22:00', 'end': '06:00', 'price_per_kWh': 0.2}
            ]},
        }

    def score(self, timestamps, powers, states=None, rooms=None, dt=1):
        n = len(timestamps)
        result = SimulationResult(
            timestamps=timestamps, outside_temperatures_C=[0] * n,
            room_temperatures_C=rooms if rooms is not None else [21] * n,
            storage_temperatures_C=[35] * n, heat_pump_on=states if states is not None else [True] * n,
            cop=[3] * n, thermal_power_W=[3 * p for p in powers], electric_power_W=powers,
            total_electric_energy_Wh=sum(powers) * dt,
            total_thermal_energy_Wh=sum(powers) * 3 * dt,
        )
        metrics = calculate_metrics(result, 1, dt, 21)
        return calculate_score(result, metrics, dt, ScoringParams.from_config(self.config))

    def test_components_count_starts_and_strict_comfort_threshold(self):
        score = self.score([f'2020-01-01 {hour:02d}:00:00' for hour in range(10, 14)],
                           [1000, 1000, 0, 1000], states=[True, True, False, True],
                           rooms=[20, 19.9, 21, 20.5])
        self.assertAlmostEqual(score.electricity_cost, 0.9)
        self.assertAlmostEqual(score.cycle_cost, 0.1)
        self.assertAlmostEqual(score.comfort_cost, .3)
        self.assertAlmostEqual(score.total_score, 1.3)

    def test_tariff_crossing_midnight_and_exact_boundaries(self):
        for timestamp, dt, expected in [
            ('2020-01-01 21:30:00', 1, 0.25),
            ('2020-01-01 05:30:00', 1, 0.25),
            ('2020-01-01 23:30:00', 1, 0.2),
            ('2020-01-01 22:00:00', 1, 0.2),
            ('2020-01-01 06:00:00', 1, 0.3),
            ('2020-01-01 00:00:00', 24, 16 * 0.3 + 8 * 0.2),
        ]:
            with self.subTest(timestamp=timestamp, dt=dt):
                self.assertAlmostEqual(self.score([timestamp], [1000], dt=dt).electricity_cost, expected)

    def test_comfort_penalty_scales_with_depth_and_duration(self):
        for room, dt, expected in [(18, 1, 6), (18, .5, 3), (19, 1, 3), (20, 1, 0), (21, 1, 0)]:
            with self.subTest(room=room, dt=dt):
                score = self.score(['2020-01-01 12:00:00'], [0], states=[False], rooms=[room], dt=dt)
                self.assertAlmostEqual(score.comfort_cost, expected)

    def test_flat_tariff_and_zero_consumption(self):
        self.config['electricity']['tariff_windows'] = []
        self.assertAlmostEqual(self.score(['2020-01-01 00:00:00'], [2000], dt=0.5).electricity_cost, 0.3)
        self.assertEqual(self.score(['2020-01-01 00:00:00'], [0], states=[False]).total_score, 0)

    def test_tariff_windows_can_be_disabled_and_reenabled(self):
        self.config['electricity']['tariff_windows_enabled'] = False
        for timestamp in ('2020-01-01 21:30:00', '2020-01-01 23:30:00', '2020-01-01 05:30:00'):
            self.assertAlmostEqual(self.score([timestamp], [1000]).electricity_cost, .3)
        self.config['electricity']['tariff_windows_enabled'] = True
        self.assertAlmostEqual(self.score(['2020-01-01 23:30:00'], [1000]).electricity_cost, .2)
        self.config['electricity']['tariff_windows_enabled'] = 'false'
        with self.assertRaises(ValueError):
            ScoringParams.from_config(self.config)

    def test_invalid_costs_and_overlapping_windows(self):
        for value in (-1, float('nan'), float('inf')):
            for key in ('cost_per_heat_pump_start', 'cost_per_comfort_degree_hour'):
                config = deepcopy(self.config)
                config[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    ScoringParams.from_config(config)
        for window in [
            {'start': '05:00', 'end': '07:00', 'price_per_kWh': 0.1},
            {'start': '12:00', 'end': '12:00', 'price_per_kWh': 0.1},
            {'start': '25:00', 'end': '13:00', 'price_per_kWh': 0.1},
            {'start': '12:00', 'end': '13:00', 'price_per_kWh': -1},
        ]:
            config = deepcopy(self.config)
            config['electricity']['tariff_windows'].append(window)
            with self.subTest(window=window), self.assertRaises(ValueError):
                ScoringParams.from_config(config)
