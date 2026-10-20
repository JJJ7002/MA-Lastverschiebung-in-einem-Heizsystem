import csv
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation.config import load_config
from support import load_test_config
from weather.loader import WeatherPoint
from run_experiments import configurations, run_experiments


def test_api_weather_prepared_once_per_station_and_study(tmp_path):
    from simulation.engine import run_simulation
    from weather.meteoswiss import load_api_weather, load_api_daily
    base = load_test_config()
    base['weather']['offline'] = False
    base['simulation'].update(start_time=None, end_time=None)
    base['control']['method'] = 'cop_optimized'
    (tmp_path / 'base.yaml').write_text(yaml.safe_dump(base), encoding='utf-8')
    settings = {'base_config': 'base.yaml', 'output_dir': 'results',
                'parameters': {'weather.station': ['BAS', 'BER'],
                               'storage.model': ['SBP 200 E', 'SBP 700 E']}}
    path = tmp_path / 'experiments.yaml'
    path.write_text(yaml.safe_dump(settings), encoding='utf-8')
    def prepare_files(config, root, granularity):
        station = config['weather']['station'].lower()
        paths = [tmp_path / f'ogd-smn_{station}_t_recent.csv',
                 tmp_path / f'ogd-smn_{station}_d_recent.csv']
        paths[0].write_text('reference_timestamp;tre200s0\n01.01.2020 00:00;0\n')
        paths[1].write_text(f'station_abbr;reference_timestamp;tre200d0\n{station.upper()};01.01.2020 00:00;0\n')
        return paths
    def simulate(config, **inputs):
        assert config['weather']['offline'] is False
        assert 'weather_points' in inputs and 'daily_temperatures' in inputs
        return run_simulation(config, **inputs)
    with patch('run_experiments.download_files', side_effect=prepare_files) as prepare, patch(
            'run_experiments.run_simulation', side_effect=simulate) as simulation, patch(
            'run_experiments.load_api_weather', wraps=load_api_weather) as observations, patch(
            'run_experiments.load_api_daily', wraps=load_api_daily) as daily, patch(
            'weather.meteoswiss.download_files', side_effect=AssertionError('Unexpected repeated file check')):
        for _ in range(2):
            study = run_experiments(path)
            assert load_config(study / 'run_000/config.yaml')['weather']['offline'] is False
        assert prepare.call_count == 4  # BAS and BER once each in each study
        assert simulation.call_count == 10
        assert observations.call_count == 4
        assert daily.call_count == 4
        assert [call.args[0]['weather']['station'] for call in prepare.call_args_list] == [
            'BAS', 'BER', 'BAS', 'BER']


def test_failed_weather_preparation_is_not_repeated(tmp_path):
    base = load_test_config()
    (tmp_path / 'base.yaml').write_text(yaml.safe_dump(base), encoding='utf-8')
    path = tmp_path / 'experiments.yaml'
    path.write_text(yaml.safe_dump({'base_config': 'base.yaml', 'output_dir': 'results',
                                   'parameters': {'storage.model': ['SBP 200 E', 'SBP 700 E']}}), encoding='utf-8')
    with patch('run_experiments.download_files', side_effect=RuntimeError('Download failed')) as prepare, patch(
            'run_experiments.run_simulation') as simulation:
        study = run_experiments(path)
    prepare.assert_called_once()
    simulation.assert_not_called()
    with (study / 'summary.csv').open(encoding='utf-8', newline='') as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert all(row['status'] == 'error' and 'Download failed' in row['error'] for row in rows)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.base = load_test_config()
        self.base['weather']['source'] = 'csv'  # Isolate model tests from network access
        self.base['simulation']['weather_file'] = 'data/weather/ogd-smn_bas_t_historical_2020-2029.csv'
        self.base['simulation'].update(start_time=None, end_time=None)
        # default.yaml has no schedule windows any more; one inactive window keeps the list-index tests meaningful
        self.base['control']['schedule']['windows'] = [{'start': '22:00', 'end': '06:00', 'mode': 'force_off'}]

    def test_grid_and_baseline_are_independent(self):
        original = deepcopy(self.base)
        cases = list(configurations(self.base, {
            'control.heating_curve.slope_factor': [0.3, 0.7],
            'control.heating_curve.temperature_hysteresis_K': [3, 8],
        }))
        self.assertEqual(len(cases), 5)
        self.assertEqual(cases[0], ('baseline', original))
        self.assertEqual(len({(c['control']['heating_curve']['slope_factor'],
                              c['control']['heating_curve']['temperature_hysteresis_K'])
                             for _, c in cases[1:]}), 4)
        cases[1][1]['building']['T_room_init_C'] = 0
        self.assertEqual(self.base, original)
        self.assertEqual(cases[2][1]['building'], original['building'])

    def test_invalid_grid_is_rejected(self):
        for grid in ({}, {'control.typo': [1]}, {'control.heating_curve.slope_factor': []},
                     {'analysis.output_dir': ['elsewhere']},
                     {'control.schedule.windows.99.start': ['10:00']},
                     {'control.schedule.windows.-1.start': ['10:00']},
                     {'storage': [300]}, {'storage..model': ['SBP 200 E']}):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                list(configurations(self.base, grid))

    def test_non_control_parameters_and_list_entries(self):
        original = deepcopy(self.base)
        cases = list(configurations(self.base, {
            'storage.model': ['SBP 200 E', 'SBP 700 E'],
            'building.T_set_C': [20, 22],
            'control.schedule.windows.0.start': ['08:00', '09:00'],
            'heat_pump.T_sink_max_C': [60],
            'simulation.end_time': [None],
        }))
        self.assertEqual(len(cases), 9)
        for _, config in cases[1:]:
            self.assertIn(config['storage']['model'], ['SBP 200 E', 'SBP 700 E'])
            self.assertIn(config['building']['T_set_C'], [20, 22])
            self.assertIn(config['control']['schedule']['windows'][0]['start'], ['08:00', '09:00'])
            self.assertEqual(config['heat_pump']['T_sink_max_C'], 60)
            self.assertIsNone(config['simulation']['end_time'])
        self.assertEqual(self.base, original)

    def test_failed_variant_does_not_stop_study_and_outputs_are_isolated(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'base.yaml').write_text(yaml.safe_dump(self.base), encoding='utf-8')
            settings = {
                'base_config': 'base.yaml', 'output_dir': 'results',
                'parameters': {'control.heating_curve.temperature_hysteresis_K': [-1, 5]},
                'save_time_series': True,
            }
            path = root / 'experiments.yaml'
            path.write_text(yaml.safe_dump(settings), encoding='utf-8')
            with patch('simulation.engine.load_weather_csv', return_value=[
                WeatherPoint(datetime(2020, 1, 1), 0)
            ]):
                study = run_experiments(path)
                second = run_experiments(path)
            self.assertNotEqual(study, second)
            with (study / 'summary.csv').open(newline='', encoding='utf-8') as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([r['status'] for r in rows], ['completed', 'error', 'completed'])
            self.assertIn('ValueError', rows[1]['error'])
            self.assertEqual(rows[1]['total_electric_energy_Wh'], '')
            self.assertNotIn('target_below_hours', rows[0])
            for removed in ('target_temperature_C', 'average_room_temperature_C',
                            'minimum_room_temperature_C', 'maximum_room_temperature_C',
                            'heat_pump_on_share', 'comfort_threshold_C', 'comfort_buffer_K', 'currency',
                            'fallback_active_hours'):
                self.assertNotIn(removed, rows[0])
            self.assertEqual(float(rows[0]['comfort_below_hours']), 0)
            with (study / 'ranking.csv').open(newline='', encoding='utf-8') as fh:
                ranked = list(csv.DictReader(fh))
            self.assertEqual(len(ranked), 2)
            self.assertEqual([r['rank'] for r in ranked], ['1', '2'])
            self.assertNotIn('run_001', [r['run_id'] for r in ranked])
            scores = [float(r['total_score']) for r in ranked]
            self.assertEqual(scores, sorted(scores))
            for row in ranked:
                self.assertAlmostEqual(float(row['total_score']), sum(float(row[key]) for key in
                    ('electricity_cost', 'cycle_cost', 'comfort_cost')))
            for run in ('run_000', 'run_002'):
                self.assertTrue((study / run / 'simulation_results.csv').exists())
                self.assertTrue((study / run / 'simulation_metrics.csv').exists())
            self.assertTrue((study / 'run_001/error.txt').exists())
            self.assertEqual(load_config(root / 'base.yaml'), self.base)

    def test_volume_and_tariff_changes_reach_simulation_and_scoring(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'base.yaml').write_text(yaml.safe_dump(self.base), encoding='utf-8')
            settings = {'base_config': 'base.yaml', 'output_dir': 'results', 'parameters': {
                'storage.model': ['SBP 200 E', 'SBP 700 E'],
                'scoring.electricity.default_price_per_kWh': [0.1, 0.4],
            }}
            path = root / 'experiments.yaml'
            path.write_text(yaml.safe_dump(settings), encoding='utf-8')
            with patch('simulation.engine.load_weather_csv', return_value=[
                WeatherPoint(datetime(2020, 1, 1, 10), -10)
            ]):
                study = run_experiments(path)
            with (study / 'summary.csv').open(newline='', encoding='utf-8') as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(row['status'] == 'completed' for row in rows))
            for row in rows[1:]:
                self.assertAlmostEqual(float(row['electricity_cost']),
                    float(row['total_electric_energy_Wh']) / 1000
                    * float(row['scoring.electricity.default_price_per_kWh']), places=7)
            temperatures = []
            volume_runs = [row['run_id'] for row in rows[1:]
                           if float(row['scoring.electricity.default_price_per_kWh']) == 0.1]
            for run_id in volume_runs:
                with (study / run_id / 'simulation_results.csv').open(newline='', encoding='utf-8') as fh:
                    temperatures.append(float(next(csv.DictReader(fh))['storage_temperature_C']))
            self.assertNotEqual(*temperatures)
