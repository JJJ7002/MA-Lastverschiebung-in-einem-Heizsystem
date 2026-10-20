from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.plots import calculate_daily_temperature_profile
from analysis.plots import calculate_daily_power_shares
from analysis.plots import calculate_daily_heat_profile
from control.temperature_targets import NightSetback


class HourlyRuntimeTests(unittest.TestCase):
    def test_daily_heat_profile_averages_each_flow_and_preserves_missing_bins(self):
        result = SimpleNamespace(
            timestamps=['2020-01-01 06:00:00', '2020-01-02 06:00:00', '2020-01-01 06:10:00'],
            heating_power_to_building_W=[0, 4000, 6000],
            thermal_power_W=[2000, 4000, 0],
            building_heat_loss_W=[1000, 2000, -500],
            storage_temperatures_C=[40, 50, 42],
        )
        rows = calculate_daily_heat_profile(result, .1666666667)
        self.assertEqual(len(rows), 144)
        self.assertEqual(rows[36]['sample_count'], 2)
        self.assertEqual(rows[36]['heating_power_to_building_W'], 2000)
        self.assertEqual(rows[36]['thermal_power_W'], 3000)
        self.assertEqual(rows[36]['building_heat_loss_W'], 1500)
        self.assertEqual(rows[37]['building_heat_loss_W'], -500)
        self.assertEqual(rows[37]['thermal_power_W'], 0)
        self.assertEqual(rows[36]['storage_temperature_C'], 45)
        self.assertEqual(rows[37]['storage_temperature_C'], 42)
        self.assertIsNone(rows[0]['storage_temperature_C'])
        self.assertIsNone(rows[0]['thermal_power_W'])
        self.assertEqual(len(calculate_daily_heat_profile(result, .5)), 48)
        for dt in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                calculate_daily_heat_profile(result, dt)
        result.thermal_power_W = []
        with self.assertRaises(ValueError):
            calculate_daily_heat_profile(result, 1)

    def test_power_profile_averages_modulation_and_off_steps(self):
        result = SimpleNamespace(
            timestamps=['2020-01-01 06:00:00', '2020-01-02 06:00:00',
                        '2020-01-01 06:10:00', '2020-01-02 06:10:00'],
            heat_pump_power_fractions=[.2, 0, 1, .5],
        )
        shares = calculate_daily_power_shares(result, .1666666667)
        self.assertEqual(len(shares), 144)
        self.assertEqual(shares[36], .1)
        self.assertEqual(shares[37], .75)
        self.assertIsNone(shares[0])
        self.assertAlmostEqual(calculate_daily_power_shares(result, 1)[6], .425)
        result.heat_pump_power_fractions = [0] * 4
        self.assertEqual(calculate_daily_power_shares(result, 1)[6], 0)
        result.heat_pump_power_fractions = [1.1] * 4
        with self.assertRaises(ValueError):
            calculate_daily_power_shares(result, 1)

    def test_temperature_profile_resolution_means_and_setpoints(self):
        result = SimpleNamespace(
            timestamps=['2020-01-01 05:50:00', '2020-01-02 05:50:00',
                        '2020-01-01 06:00:00', '2020-01-01 08:00:00'],
            room_temperatures_C=[18, 20, 19, 20],
            storage_temperatures_C=[30, 34, 35, 40],
            target_room_temperature_C=21, comfort_buffer_K=1,
            night_setback=NightSetback(enabled=True),
        )
        rows = calculate_daily_temperature_profile(result, .1666666667)
        self.assertEqual(len(rows), 144)
        self.assertEqual(rows[35]['sample_count'], 2)
        self.assertEqual(rows[35]['room_temperature_C'], 19)
        self.assertEqual(rows[35]['room_temperature_min_C'], 18)
        self.assertEqual(rows[35]['room_temperature_max_C'], 20)
        self.assertEqual(rows[36]['room_temperature_min_C'], 19)
        self.assertEqual(rows[36]['room_temperature_max_C'], 19)
        self.assertIsNone(rows[0]['room_temperature_min_C'])
        self.assertIsNone(rows[0]['room_temperature_max_C'])
        self.assertEqual(rows[35]['storage_temperature_C'], 32)
        self.assertIsNone(rows[0]['storage_temperature_C'])
        self.assertAlmostEqual(rows[35]['target_room_temperature_C'], 21 - 211/20736)
        self.assertEqual(rows[36]['target_room_temperature_C'], 21)
        self.assertEqual(rows[35]['comfort_threshold_C'], 18)
        self.assertEqual(rows[36]['comfort_threshold_C'], 20)
        self.assertEqual(rows[48]['comfort_threshold_C'], 20)
        self.assertIsNone(rows[0]['room_temperature_C'])
        self.assertEqual(len(calculate_daily_temperature_profile(result, .5)), 48)
        for dt in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                calculate_daily_temperature_profile(result, dt)
