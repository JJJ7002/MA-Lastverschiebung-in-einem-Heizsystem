"""Room feedback gain (heating.room_response_minutes)."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support import load_test_config
from simulation.engine import build_simulation_params


def configuration(minutes=2):
    config = load_test_config()
    config["building"]["T_mass_init_C"] = 20.9
    config['simulation']['dt_minutes'] = minutes
    config['control']['night_setback']['enabled'] = False
    config['building']['T_room_init_C'] = 20.9
    config['storage']['T_storage_init_C'] = 60
    config['heat_pump']['min_off_time_min'] = 1000
    config['heating']['radiator']['reference_power_W'] = 30000
    return config


def test_response_defaults_and_configured_gain():
    config = configuration()
    config['heating'].pop('room_response_minutes')
    params = build_simulation_params(config)
    assert params.room_response_minutes == 30
    # Loss part without room error; the error part scales with 1 / response time.
    base = params.room_heat_demand_W(20, 0, 20, 20)
    fast = params.room_heat_demand_W(20, 0, 21, 20)
    assert fast > base
    config['heating']['room_response_minutes'] = 60
    slow = build_simulation_params(config).room_heat_demand_W(20, 0, 21, 20)
    assert fast - base == pytest.approx(2 * (slow - base))
    # Room clearly above target and no coupling to the mass: no heat requested.
    assert params.room_heat_demand_W(25, 20, 21, 25) == 0


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_invalid_response_rejected(value):
    config = configuration()
    config['heating']['room_response_minutes'] = value
    with pytest.raises(ValueError, match='room_response_minutes'):
        build_simulation_params(config)
