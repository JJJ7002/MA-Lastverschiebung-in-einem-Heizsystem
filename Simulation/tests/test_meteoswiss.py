from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock
from urllib.error import HTTPError, URLError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weather.meteoswiss import cached_get, download_files, load_api_weather, load_api_daily
from weather.loader import load_weather_csv
from support import load_test_config
from simulation.engine import run_simulation


def config():
    return {'weather': {'source': 'meteoswiss', 'station': 'BAS', 'cache_dir': 'cache'},
            'simulation': {'start_time': '2020-01-01 00:04', 'end_time': '2020-01-01 00:14'}}


def test_discovery_downloads_full_history_and_offline_cache(tmp_path):
    settings = config()
    settings['weather']['offline'] = True
    folder = tmp_path / 'cache/bas'
    folder.mkdir(parents=True)
    names = ['ogd-smn_bas_t_historical_2010-2019.csv',
             'ogd-smn_bas_t_historical_2020-2029.csv', 'ogd-smn_bas_t_recent.csv']
    (folder / 'catalog.json').write_text(json.dumps({'assets': {
        name: {'href': 'https://example.test/' + name} for name in names}}))
    (folder / names[0]).write_text('previous')
    (folder / names[1]).write_text('current')
    (folder / names[2]).write_text('recent')
    with patch('weather.meteoswiss.urlopen') as network:
        paths = download_files(settings, tmp_path)
        network.assert_not_called()
    assert [path.name for path in paths] == names


def test_station_switch_and_full_archive_are_checked_each_start(tmp_path):
    settings = config()
    settings['weather']['station'] = 'BER'
    names = ['ogd-smn_ber_t_historical_1980-1989.csv',
             'ogd-smn_ber_t_historical_2020-2029.csv', 'ogd-smn_ber_t_recent.csv',
             'ogd-smn_ber_d_historical.csv', 'ogd-smn_ber_d_recent.csv']
    excluded = ['ogd-smn_ber_t_now.csv', 'ogd-smn_ber_d_now.csv']
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(json.dumps({'assets': {name: {'href': 'https://example.test/' + name}
                                                for name in names + excluded}}))
    def fetch(url, path, **kwargs):
        assert kwargs['max_age'] == 0
        assert Path(path).parent.name == 'ber'
        return catalog if str(path).endswith('catalog.json') else Path(path)
    with patch('weather.meteoswiss.cached_get', side_effect=fetch) as network:
        paths = download_files(settings, tmp_path, None)
        download_files(settings, tmp_path, None)
    assert {p.name for p in paths} == set(names)
    assert network.call_count == 2 * (len(names) + 1)


def test_conditional_download_and_network_fallback(tmp_path):
    path = tmp_path / 'weather.csv'
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = b'original data'
    response.headers = {'ETag': 'v1'}
    with patch('weather.meteoswiss.urlopen', return_value=response):
        cached_get('https://example.test/weather', path)
    assert path.read_bytes() == b'original data'
    with patch('weather.meteoswiss.urlopen', side_effect=HTTPError('url', 304, '', {}, None)) as network:
        cached_get('https://example.test/weather', path, max_age=-1)
        assert network.call_args.args[0].get_header('If-none-match') == 'v1'
    assert path.read_bytes() == b'original data'
    response.read.return_value = b'updated data'
    response.headers = {'ETag': 'v2'}
    with patch('weather.meteoswiss.urlopen', return_value=response):
        cached_get('https://example.test/weather', path, max_age=-1)
    assert path.read_bytes() == b'updated data'
    assert path.with_suffix('.csv.etag').read_text() == 'v2'
    with patch('weather.meteoswiss.urlopen', side_effect=URLError('offline')), pytest.warns(UserWarning):
        cached_get('https://example.test/weather', path, max_age=-1)
    with pytest.raises(FileNotFoundError):
        cached_get('url', tmp_path / 'absent', offline=True)


def test_encoding_original_utc_and_simulation_subinterval(tmp_path):
    path = tmp_path / 'data.csv'
    path.write_bytes(('station_abbr;reference_timestamp;tre200s0;comment\n'
                      'BAS;01.01.2020 00:00;-5;Zürich\n'
                      'BAS;01.01.2020 00:10;-4;\n').encode('cp1252'))
    assert load_weather_csv(path)[0].timestamp == datetime(2020, 1, 1)
    settings = config()
    with patch('weather.meteoswiss.download_files', return_value=[path]):
        weather = load_api_weather(settings, tmp_path)
    full = load_test_config()
    full['simulation'].update(settings['simulation'])
    full['control']['method'] = 'heating_curve'
    with patch('simulation.engine.load_api_weather', return_value=weather):
        result = run_simulation(full)
    assert len(result.timestamps) == 5
    assert '00:04' in result.timestamps[0]
    assert '00:12' in result.timestamps[-1]
    full['simulation']['end_time'] = '2020-01-01 00:06'
    with patch('weather.meteoswiss.download_files', return_value=[path]):
        result = run_simulation(full)
    assert len(result.timestamps) == 1
    assert '00:04' in result.timestamps[0]


def test_daily_station_and_newer_file_wins(tmp_path):
    paths = [tmp_path / 'historical.csv', tmp_path / 'recent.csv']
    for path, value in zip(paths, (1, 2)):
        path.write_text('station_abbr;reference_timestamp;tre200d0\n'
                        f'BER;01.01.2020 00:00;{value}\nBAS;01.01.2020 00:00;99\n')
    settings = config()
    settings['weather']['station'] = 'BER'
    with patch('weather.meteoswiss.download_files', return_value=paths):
        daily = load_api_daily(settings, tmp_path)
    assert daily == {datetime(2020, 1, 1).date(): 2}


def test_missing_measurements_fail(tmp_path):
    path = tmp_path / 'data.csv'
    path.write_text('reference_timestamp;tre200s0\n01.01.2020 00:00;0\n01.01.2020 00:20;1\n')
    settings = config()
    settings['simulation']['end_time'] = '2020-01-01 00:30'
    settings['weather']['fill_gaps'] = False
    with patch('weather.meteoswiss.download_files', return_value=[path]), pytest.raises(ValueError, match='Missing'):
        load_api_weather(settings, tmp_path)


def _gap_csv(tmp_path, days=5, gap=(datetime(2020, 1, 3, 6, 0), datetime(2020, 1, 3, 9, 0))):
    from datetime import timedelta
    lines = ['reference_timestamp;tre200s0;gre000z0']
    t = datetime(2020, 1, 1)
    while t < datetime(2020, 1, 1) + timedelta(days=days):
        if not (gap[0] <= t < gap[1]):
            temperature = 5 + 3 * ((t.hour * 60 + t.minute) / 1440) + t.day  # daily profile + day offset
            lines.append(f'{t:%d.%m.%Y %H:%M};{temperature};300')
        t += timedelta(minutes=10)
    path = tmp_path / 'gap.csv'
    path.write_text('\n'.join(lines) + '\n')
    return path


def test_gaps_are_filled_with_neighbour_days_and_reported(tmp_path, capsys):
    path = _gap_csv(tmp_path)
    settings = config()
    settings['simulation'].update(start_time='2020-01-01 00:00', end_time='2020-01-06 00:00')
    with patch('weather.meteoswiss.download_files', return_value=[path]):
        points = load_api_weather(settings, tmp_path)
    stamps = [p.timestamp for p in points]
    assert all(b - a == timedelta(minutes=10) for a, b in zip(stamps, stamps[1:]))
    filled = [p for p in points if datetime(2020, 1, 3, 6, 0) <= p.timestamp < datetime(2020, 1, 3, 9, 0)]
    assert len(filled) == 18 and all(p.global_radiation_W_m2 == 0 for p in filled)
    # neighbour days differ by +-1 K from day 3, the mean equals day 3 exactly
    expected = 5 + 3 * (7.5 * 60 / 1440) + 3
    middle = next(p for p in filled if p.timestamp == datetime(2020, 1, 3, 7, 30))
    assert middle.outside_temperature_C == pytest.approx(expected, abs=1e-6)
    out = capsys.readouterr().out
    assert '1 Lücke' in out and '3.0 h' in out and '2020-01-03 06:00' in out


def test_complete_data_is_reported_and_too_long_gaps_fail(tmp_path, capsys):
    settings = config()
    settings['simulation'].update(start_time='2020-01-01 00:00', end_time='2020-01-06 00:00')
    complete = _gap_csv(tmp_path, gap=(datetime(2030, 1, 1), datetime(2030, 1, 2)))
    with patch('weather.meteoswiss.download_files', return_value=[complete]):
        load_api_weather(settings, tmp_path)
    assert 'vollständig' in capsys.readouterr().out
    long_gap = _gap_csv(tmp_path)
    settings['weather']['max_gap_hours'] = 2
    with patch('weather.meteoswiss.download_files', return_value=[long_gap]):
        with pytest.raises(ValueError, match='max_gap_hours'):
            load_api_weather(settings, tmp_path)
        assert len(load_api_weather(settings, tmp_path, allow_gaps=True)) < 5 * 144
