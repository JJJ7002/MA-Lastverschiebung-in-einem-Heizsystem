"""Interpolation of exported WPL 25 A chart data, without network access."""
import csv
from bisect import bisect_left
from functools import lru_cache
from math import isfinite
from pathlib import Path

import numpy as np


def bracket(values, x):
    x = min(max(x, values[0]), values[-1])
    right = min(max(bisect_left(values, x), 1), len(values) - 1)
    left = right - 1
    return values[left], values[right], (x - values[left]) / (values[right] - values[left])


def _bracket_index(values, x):
    """Like bracket(), but returns the neighbouring indices instead of the values."""
    x = min(max(x, values[0]), values[-1])
    right = min(max(bisect_left(values, x), 1), len(values) - 1)
    left = right - 1
    return left, right, (x - values[left]) / (values[right] - values[left])


class PerformanceMap:
    def __init__(self, path):
        with Path(path).open(encoding="utf-8", newline="") as file:
            rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(file)]
        self.grid = {(r['source_temp_C'], r['sink_temp_C'], r['inverter_speed_percent']):
                     (r['thermal_power_W'], r['cop']) for r in rows}
        self.sources = sorted({key[0] for key in self.grid})
        self.sinks = sorted({key[1] for key in self.grid})
        self.speeds = sorted({key[2] for key in self.grid})
        low, high = self.speeds[0], self.speeds[-1]
        if high <= low:
            raise ValueError('Performance map must contain at least two inverter speeds')
        self.power_fractions = tuple((speed - low) / (high - low) for speed in self.speeds)
        if len(self.grid) != len(rows) or len(rows) != len(self.sources)*len(self.sinks)*len(self.speeds):
            raise ValueError('Performance map must be a complete grid without duplicates')
        if any(not isfinite(v) or v <= 0 for pair in self.grid.values() for v in pair):
            raise ValueError('Performance map requires positive finite power and COP')

    def _dense_table(self):
        # Dense [source, sink, (power, cop), speed] array holding the same values as self.grid.
        table = self.__dict__.get('_table')
        if table is None:
            table = np.empty((len(self.sources), len(self.sinks), 2, len(self.speeds)))
            for i, source in enumerate(self.sources):
                for j, sink in enumerate(self.sinks):
                    for k, speed in enumerate(self.speeds):
                        table[i, j, :, k] = self.grid[source, sink, speed]
            self.__dict__['_table'] = table
        return table

    def max_electric_power_W(self):
        """Largest electrical input over the manufacturer operating range (-20..40 C source)."""
        value = self.__dict__.get('_max_electric_W')
        if value is None:
            value = max(power / cop for (source, _, _), (power, cop) in self.grid.items()
                        if -20.0 <= source <= 40.0)
            self.__dict__['_max_electric_W'] = value
        return value

    @lru_cache(maxsize=4096)
    def curves(self, source, sink):
        # Manufacturer operating range, not the chart's padded -40 C range.
        source = min(max(source, -20.0), 40.0)
        ia, ib, u = _bracket_index(self.sources, source)
        ic, id_, v = _bracket_index(self.sinks, sink)
        table = self._dense_table()
        # Same bilinear expression and operation order as the scalar form, for all speeds at once.
        power, cop = ((1-v)*((1-u)*table[ia, ic]+u*table[ib, ic])
                      + v*((1-u)*table[ia, id_]+u*table[ib, id_])).tolist()
        return tuple(zip(power, cop))

    @lru_cache(maxsize=16384)  # Several evaluations per step share identical arguments.
    def performance(self, source, sink, fraction=1.0):
        if not isfinite(fraction) or not (fraction == -1 or 0 <= fraction <= 1):
            raise ValueError('power_fraction must be -1 (off) or in [0, 1] (minimum to maximum)')
        if fraction == -1:
            return 0.0, 0.0, 0.0
        curve = self.curves(source, sink)
        speed = self.speeds[0] + fraction * (self.speeds[-1] - self.speeds[0])
        right = min(max(bisect_left(self.speeds, speed), 1), len(self.speeds) - 1)
        left = right - 1
        t = (speed - self.speeds[left]) / (self.speeds[right] - self.speeds[left])
        thermal = curve[left][0] + t * (curve[right][0] - curve[left][0])
        cop = curve[left][1] + t * (curve[right][1] - curve[left][1])
        return cop, thermal, thermal / cop


@lru_cache(maxsize=8)
def load_performance_map(path):
    return PerformanceMap(path)
