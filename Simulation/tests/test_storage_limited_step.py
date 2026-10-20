"""Combined flux evaluation preserves the original adaptive integration."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.stratified_tank import StorageParameters, StratifiedStorage, PortFlow


@pytest.mark.parametrize('maximum', [0.1, 1000.0])
def test_limited_step_matches_separate_limit_and_step(maximum):
    tank = StratifiedStorage(StorageParameters(415, 10, 0.6, 1.48))
    temperatures = np.linspace(60, 30, 10)
    ports = [PortFlow(1, 1., 65.), PortFlow(9, -1.),
             PortFlow(0, -0.3), PortFlow(8, 0.3, 32.)]
    expected_dt = min(maximum, tank.max_step_seconds(temperatures, ports, 21.))
    expected = tank.step(temperatures, ports, 21., expected_dt)
    actual_dt, actual = tank.limited_step(temperatures, ports, 21., maximum)
    assert actual_dt == expected_dt
    np.testing.assert_array_equal(actual.temperatures_C, expected.temperatures_C)
    for name in ('port_heat_W', 'loss_W', 'energy_change_Wh', 'balance_error_Wh'):
        assert getattr(actual, name) == getattr(expected, name)


def test_storage_step_is_explicit_and_enforces_cfl_limit():
    tank = StratifiedStorage(StorageParameters(415, 10, 0.63, 1.48))
    temperatures = np.linspace(60, 30, 10)
    ports = [PortFlow(1, 0.8, 65.), PortFlow(9, -0.8)]
    rates, _, _, _ = tank._rates(temperatures, ports, 21.)
    dt = tank.max_step_seconds(temperatures, ports, 21.)

    result = tank.step(temperatures, ports, 21., dt * 0.8)
    expected = tank.mix_inversions(
        temperatures + rates * (dt * 0.8) / tank.capacities_J_K)
    np.testing.assert_allclose(result.temperatures_C, expected, rtol=0, atol=1e-12)

    with pytest.raises(ValueError, match='CFL limit'):
        tank.step(temperatures, ports, 21., dt * 1.01)
