"""Shared test configuration.

The tests use config/default.yaml as plant description, but this file also
contains run settings that are tuned for real studies (multi-year period,
48 h settling, summer shutdown).  Tests drive the simulation with their own
short weather series and arbitrary dates, so those run settings are
neutralised here instead of in every single test.  The summer shutdown and
the settling phase have their own explicit tests.
"""
from __future__ import annotations

from pathlib import Path

from simulation.config import load_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'config/default.yaml'


def load_test_config():
    config = load_config(DEFAULT_CONFIG)
    simulation = config['simulation']
    simulation['settling_hours'] = 0.0
    simulation['start_time'] = None
    simulation['end_time'] = None
    schedule = config['control'].setdefault('schedule', {})
    schedule.setdefault('seasonal_heating_off', {})['enabled'] = False
    return config
