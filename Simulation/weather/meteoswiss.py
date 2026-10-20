"""MeteoSwiss station CSV discovery and conditional downloads (original UTC)."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import warnings

API = 'https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-smn/items/'


def cached_get(url, path, *, offline=False, max_age=3600):
    path = Path(path)
    if path.exists() and (offline or datetime.now().timestamp() - path.stat().st_mtime < max_age):
        return path
    if offline:
        raise FileNotFoundError(f'Weather cache missing: {path}; start a simulation online first')
    path.parent.mkdir(parents=True, exist_ok=True)
    tag_path = path.with_suffix(path.suffix + '.etag')
    headers = {'User-Agent': 'MA-weather-loader/1.0'}
    if path.exists() and tag_path.exists():
        headers['If-None-Match'] = tag_path.read_text(encoding='utf-8')
    try:
        with urlopen(Request(url, headers=headers), timeout=60) as response:
            content = response.read()
            tag = response.headers.get('ETag', '')
        temporary = path.with_suffix(path.suffix + '.part')
        temporary.write_bytes(content)
        temporary.replace(path)
        tag_path.write_text(tag, encoding='utf-8')
    except HTTPError as error:
        if error.code == 304 and path.exists():
            path.touch()
        else:
            raise RuntimeError(f'MeteoSwiss download failed: {url}: {error}') from error
    except (URLError, TimeoutError) as error:
        if not path.exists():
            raise RuntimeError(f'MeteoSwiss unavailable and no cached file: {url}') from error
        warnings.warn(f'MeteoSwiss unavailable; using cached data: {path}', stacklevel=2)
    return path


def parse_time(value):
    if value is None or value == '':
        return None
    if isinstance(value, datetime):
        result = value
    else:
        result = None
        for fmt in ('%Y-%m-%d %H:%M', '%Y-%m-%d %H:%M:%S', '%d.%m.%Y %H:%M', '%Y-%m-%d'):
            try:
                result = datetime.strptime(str(value), fmt)
                break
            except ValueError:
                pass
        if result is None:
            raise ValueError(f'Unsupported UTC datetime: {value}')
    if result.tzinfo is not None:
        raise ValueError('Weather dates must use original UTC without a timezone suffix')
    return result


def download_files(config, project_root, granularity='t'):
    """Sync historical and recent files, independent of the simulation window.

    granularity=None syncs both ten-minute observations and official daily means.
    Every online call checks the catalog and file ETags.
    """
    settings = config.get('weather', {})
    station = str(settings.get('station', 'BAS')).strip().lower()
    if not re.fullmatch('[a-z]{3}', station):
        raise ValueError('weather.station must be a three-letter station code')
    cache = Path(project_root) / settings.get('cache_dir', 'weather_cache') / station
    offline = bool(settings.get('offline', False))
    item = json.loads(cached_get(API + station, cache / 'catalog.json', offline=offline,
                                 max_age=0).read_text(encoding='utf-8'))
    selected = []
    granularities = ('t', 'd') if granularity is None else (granularity,)
    for name, asset in item['assets'].items():
        match = re.fullmatch(rf'ogd-smn_{station}_([td])_(historical|recent)(?:_(\d{{4}})-(\d{{4}}))?\.csv', name)
        if match and match.group(1) in granularities:
            selected.append(({'historical': 0, 'recent': 1}[match.group(2)], name, asset['href']))
    for value in granularities:
        if not any(f'_{value}_' in name for _, name, _ in selected):
            raise ValueError(f'No MeteoSwiss {value} files for {station.upper()}')
    print(f"Wetterdaten {station.upper()}: vollständigen Datenbestand prüfen" +
          (' (offline).' if offline else '.'))
    return [cached_get(url, cache / name, offline=offline, max_age=0)
            for _, name, url in sorted(selected)]


def weather_source(config):
    source = config.get('weather', {}).get(
        'source', 'csv' if 'weather_file' in config.get('simulation', {}) else 'meteoswiss')
    if source not in ('csv', 'meteoswiss'):
        raise ValueError('weather.source must be csv or meteoswiss')
    return source


def fill_weather_gaps(points, start, end, max_gap_hours, *, search_days=7):
    """Fill short measurement gaps with the daily profile of neighbouring days.

    For every missing 10-minute time the temperature of the same clock time on the nearest earlier
    and later day with a measurement is averaged; the difference between this profile and the
    measured values at both gap borders is faded out linearly, so the series joins up without a
    jump. Without usable neighbour days the borders are connected linearly. Global radiation is 0
    in filled intervals. Gaps longer than ``max_gap_hours`` stay open and are returned separately.
    Returns (points, [(first filled, last filled, hours)], [(border before, border after)]).
    """
    step = timedelta(minutes=10)
    measured = {p.timestamp: p.outside_temperature_C for p in points}

    def profile(moment):
        values = []
        for sign in (-1, 1):
            for day in range(1, search_days + 1):
                value = measured.get(moment + sign * timedelta(days=day))
                if value is not None:
                    values.append(value)
                    break
        return sum(values) / len(values) if values else None

    result, filled, skipped = list(points), [], []
    for a, b in zip(points, points[1:]):
        if (b.timestamp - a.timestamp <= step
                or not (start is None or b.timestamp > start)
                or not (end is None or a.timestamp < end - step)):
            continue
        missing = (b.timestamp - a.timestamp) / step - 1
        if missing * step > timedelta(hours=max_gap_hours):
            skipped.append((a.timestamp, b.timestamp))
            continue
        offset_a = offset_b = None
        profile_a, profile_b = profile(a.timestamp), profile(b.timestamp)
        if profile_a is not None and profile_b is not None:
            offset_a, offset_b = a.outside_temperature_C - profile_a, b.outside_temperature_C - profile_b
        span = (b.timestamp - a.timestamp).total_seconds()
        moment = a.timestamp + step
        while moment < b.timestamp:
            fraction = (moment - a.timestamp).total_seconds() / span
            base = profile(moment)
            if base is None or offset_a is None:
                temperature = a.outside_temperature_C + fraction * (b.outside_temperature_C - a.outside_temperature_C)
            else:
                temperature = base + offset_a + fraction * (offset_b - offset_a)
            result.append(type(a)(moment, temperature, 0.0))
            moment += step
        filled.append((a.timestamp + step, b.timestamp - step, missing * step.total_seconds() / 3600))
    result.sort(key=lambda point: point.timestamp)
    return result, filled, skipped


def report_weather_gaps(station, filled):
    if not filled:
        print(f'Wetterdaten {station}: Datenbestand vollständig, keine Lücken.')
        return
    total = sum(hours for _, _, hours in filled)
    print(f'Wetterdaten {station}: {len(filled)} Lücke(n) mit insgesamt {total:.1f} h aufgefüllt '
          f'(Temperatur: Tagesverlauf der Nachbartage, Strahlung: 0):')
    for first, last, hours in filled:
        print(f'  {first:%Y-%m-%d %H:%M} bis {last:%Y-%m-%d %H:%M} UTC ({hours:.1f} h)')


def load_api_weather(config, project_root, *, files=None, allow_gaps=False):
    from weather.loader import load_weather_csv
    start = parse_time(config['simulation'].get('start_time'))
    end = parse_time(config['simulation'].get('end_time'))
    if start and end and start >= end:
        raise ValueError('Simulation start must be before end (exclusive)')
    merged = {}
    for path in (download_files(config, project_root, None) if files is None else files):
        if '_d_' in path.name:
            continue
        block = re.search(r'_historical_(\d{4})-(\d{4})', path.name)
        if block:
            if start and datetime(int(block[2]) + 1, 1, 1) <= start - timedelta(minutes=10):
                continue
            if end and datetime(int(block[1]), 1, 1) >= end + timedelta(minutes=10):
                continue
        for point in load_weather_csv(path,
                                      start=start - timedelta(minutes=10) if start else None,
                                      end=end + timedelta(minutes=10) if end else None):
            merged[point.timestamp] = point
    points = [merged[key] for key in sorted(merged)]
    if not points:
        raise ValueError('No MeteoSwiss measurements in the selected UTC window')
    settings = config.get('weather', {})
    station = str(settings.get('station', 'BAS')).upper()
    step = timedelta(minutes=10)
    filled_gaps = []
    if settings.get('fill_gaps', True):
        max_hours = float(settings.get('max_gap_hours', 48))
        points, filled_gaps, skipped = fill_weather_gaps(points, start, end, max_hours)
        if skipped and not allow_gaps:
            a, b = skipped[0]
            raise ValueError(f'Missing 10-minute weather measurements between {a} and {b} '
                             f'(longer than weather.max_gap_hours={max_hours:g})')
    for a, b in zip(points, points[1:]):
        if (not allow_gaps and b.timestamp - a.timestamp != step
                and (start is None or b.timestamp > start)
                and (end is None or a.timestamp < end - timedelta(minutes=10))):
            raise ValueError(f'Missing 10-minute weather measurements between {a.timestamp} and {b.timestamp}')
    report_weather_gaps(station, filled_gaps)
    return points


def load_api_daily(config, project_root, *, files=None):
    from weather.loader import load_daily_temperatures
    merged = {}
    for path in (download_files(config, project_root, 'd') if files is None else files):
        if files is not None and '_d_' not in path.name:
            continue
        merged.update(load_daily_temperatures(path, station=config.get('weather', {}).get('station', 'BAS')))
    return merged
