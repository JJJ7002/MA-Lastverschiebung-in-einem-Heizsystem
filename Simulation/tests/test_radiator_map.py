from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from models.radiator import RadiatorMap
from models.stratified_tank import CP_J_KG_K


def test_reference_and_room_shift():
    radiator = RadiatorMap()
    supply, ret = radiator.temperatures(6000)
    assert supply == pytest.approx(26)
    assert ret == pytest.approx(26 - 6000 / (1500 / 3600 * 4182))
    assert radiator.flow(60, 20, 6000).supply_C == pytest.approx(46)


@pytest.mark.parametrize('hot', [15, 21, 23, 35, 48, 65])
@pytest.mark.parametrize('demand', [0, 1, 1000, 6000, 15000])
def test_mixer_and_energy(hot, demand):
    radiator = RadiatorMap()
    flow = radiator.flow(hot, 21, demand)
    assert 0 <= flow.mixer_fraction <= 1 + 1e-12
    assert 0 <= flow.heat_W <= demand
    assert flow.heat_W == pytest.approx(flow.circulation_kg_s * CP_J_KG_K * (flow.supply_C - flow.return_C))
    assert flow.heat_W == pytest.approx(flow.storage_kg_s * CP_J_KG_K * (hot - flow.return_C))
    if hot >= flow.target_supply_C:
        assert flow.heat_W == pytest.approx(demand)
    elif flow.heat_W > 0:
        assert flow.supply_C == pytest.approx(hot)
        assert flow.mixer_fraction == pytest.approx(1)


@pytest.mark.parametrize('kwargs', [dict(exponent=0), dict(volume_flow_L_h=0), dict(reference_supply_C=23), dict(reference_power_W=float('nan'))])
def test_invalid_reference(kwargs):
    with pytest.raises(ValueError):
        RadiatorMap(**kwargs)


def test_inverse_and_characteristic():
    radiator = RadiatorMap()
    assert np.all(np.diff(radiator.supply_K) > 0)
    for power in [10, 999, 6000, 12000]:
        flow = radiator.flow(80, 22, power)
        a, b = flow.supply_C - 22, flow.return_C - 22
        mean = (a-b)/np.log(a/b)
        assert 6000 * (mean / radiator.reference_log_mean_K)**1.3 == pytest.approx(power, rel=0.002)


def test_default_configuration_uses_map_and_ignores_legacy_circuit_curve():
    from support import load_test_config
    from simulation.engine import build_simulation_params, initial_state
    from simulation.step import simulate_step, SimulationInput
    config = load_test_config()
    config['control']['night_setback']['enabled'] = False
    params = build_simulation_params(config)
    assert params.radiator is not None
    radiator = config['heating']['radiator']  # at its reference point the map returns the reference supply
    flow = params.heating_flow(60, radiator['reference_room_C'], radiator['reference_power_W'], 0)
    assert flow.supply_C == pytest.approx(radiator['reference_supply_C'])
    assert flow.heat_W == radiator['reference_power_W']
    state = initial_state(config, 0)
    for _ in range(10):
        result = simulate_step(state, SimulationInput(0), params)
        assert abs(result.storage_balance_error_Wh) < 1e-6
        state = result.state
