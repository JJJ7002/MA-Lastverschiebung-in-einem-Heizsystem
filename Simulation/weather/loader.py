from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
from math import isfinite, isclose


def parse_dmy_hm(text: str) -> datetime:
    """Parse 'DD.MM.YYYY HH:MM'; strptime is ~10x slower for 350k rows."""
    if len(text) == 16 and text[2] == '.' and text[5] == '.' and text[10] == ' ' and text[13] == ':':
        try:
            return datetime(int(text[6:10]), int(text[3:5]), int(text[0:2]),
                            int(text[11:13]), int(text[14:16]))
        except ValueError:
            pass
    return datetime.strptime(text, "%d.%m.%Y %H:%M")


@dataclass(frozen=True)
class WeatherPoint:
    timestamp: datetime
    outside_temperature_C: float
    global_radiation_W_m2: float = 0.0  # gre000z0, horizontal plane; 0 if not available


def resample_weather(points: Iterable[WeatherPoint], dt_hours: float) -> list[WeatherPoint]:
    """Subdivide weather intervals, holding each measurement until the next one.

    Infer the source interval from the smallest spacing; retain gaps instead of
    inventing measurements. The final measurement covers one source interval.
    A single point covers one simulation step because its cadence is unknown.
    """
    if not isfinite(dt_hours) or dt_hours <= 0:
        raise ValueError('simulation.dt_minutes must be finite and positive')
    points = list(points)
    if len(points) < 2:
        return points
    source_seconds = min((b.timestamp - a.timestamp).total_seconds()
                         for a, b in zip(points, points[1:]))
    ratio = source_seconds / (dt_hours * 3600)
    if ratio < 1 - 1e-8 or not isclose(ratio, round(ratio), abs_tol=1e-8):
        raise ValueError('Weather interval must be a positive integer multiple of simulation.dt_minutes')
    count = round(ratio)
    if count == 1:
        return points
    return [WeatherPoint(point.timestamp + timedelta(seconds=i * source_seconds / count),
                         point.outside_temperature_C, point.global_radiation_W_m2)
            for point in points for i in range(count)]


_WEATHER_CACHE: dict = {}


def load_weather_csv(path: str | Path, *, start=None, end=None) -> list[WeatherPoint]:
    """Load weather data with columns `reference_timestamp` and `tre200s0`.

    Results are cached per process, keyed by file identity (mtime and size)
    and time window, so repeated runs on the same archive do not re-parse it.
    """
    weather_path = Path(path)
    try:
        info = weather_path.stat()
        key = (str(weather_path.resolve()), info.st_mtime_ns, info.st_size, start, end)
    except OSError:
        key = None
    if key is not None and key in _WEATHER_CACHE:
        return list(_WEATHER_CACHE[key])
    points = _read_weather_csv(weather_path, start, end)
    if key is not None:
        if len(_WEATHER_CACHE) >= 16:
            _WEATHER_CACHE.pop(next(iter(_WEATHER_CACHE)))
        _WEATHER_CACHE[key] = tuple(points)
    return points


def _read_weather_csv(weather_path: Path, start, end) -> list[WeatherPoint]:
    points: list[WeatherPoint] = []

    with weather_path.open("r", encoding=_csv_encoding(weather_path), newline="") as fh:
        reader = csv.reader(fh, delimiter=";")
        header = next(reader, [])
        if not {'reference_timestamp', 'tre200s0'}.issubset(header):
            raise ValueError(f'Weather requires reference_timestamp and tre200s0: {weather_path}')
        time_column, temp_column = header.index('reference_timestamp'), header.index('tre200s0')
        radiation_column = header.index('gre000z0') if 'gre000z0' in header else None
        for row in reader:
            if not row:
                continue
            temp_raw = row[temp_column].strip()
            if not temp_raw:
                continue
            timestamp = parse_dmy_hm(row[time_column].strip())
            if (start is not None and timestamp < start) or (end is not None and timestamp >= end):
                continue
            temperature = float(temp_raw.replace(",", "."))
            if not isfinite(temperature):
                raise ValueError(f'Invalid temperature for {timestamp}')
            radiation = 0.0
            if radiation_column is not None and len(row) > radiation_column and row[radiation_column].strip():
                radiation = max(0.0, float(row[radiation_column].replace(",", ".")))
                if not isfinite(radiation):
                    raise ValueError(f'Invalid global radiation for {timestamp}')
            points.append(WeatherPoint(timestamp=timestamp, outside_temperature_C=temperature,
                                       global_radiation_W_m2=radiation))

    return points


def _csv_encoding(path):
    with Path(path).open('rb') as fh:
        return 'utf-8-sig' if fh.read(3) == b'\xef\xbb\xbf' else 'cp1252'


def load_daily_temperatures(path: str | Path, station='BAS') -> dict:
    """Read official UTC daily means; additional measurement columns are ignored."""
    temperatures = {}
    with Path(path).open('r', encoding=_csv_encoding(path), newline='') as fh:
        reader = csv.DictReader(fh, delimiter=';')
        if not {'reference_timestamp', 'tre200d0'}.issubset(reader.fieldnames or []):
            raise ValueError('Daily weather requires reference_timestamp and tre200d0')
        for row in reader:
            if row.get('station_abbr', station).strip().upper() != station.upper():
                continue
            raw = row['tre200d0'].strip()
            if not raw:
                continue
            day = parse_dmy_hm(row['reference_timestamp'].strip()).date()
            value = float(raw.replace(',', '.'))
            if not isfinite(value):
                raise ValueError(f'Invalid daily temperature for {day}')
            if day in temperatures:
                raise ValueError(f'Duplicate daily temperature for {day}')
            temperatures[day] = value
    return temperatures
