from dataclasses import replace
from types import SimpleNamespace

import pytest

from test_buffer_load import LOAD_W, configuration
from control.heat_pump_power import heat_pump_power_request
from control.operating_constraints import OperatingState
from models.storage import StorageState
from models.hydraulics import fixed_spread_heat_pump
from simulation.engine import build_simulation_params, initial_state
from simulation.step import SimulationInput, simulate_step


@pytest.mark.parametrize('on,normal_W,expected_run', [
    (False, 4999, False), (False, 5000, False), (False, 5001, True),
    (True, 999, False), (True, 1000, True), (True, 1001, True),
    (True, 4999, True), (True, 6000, True),
])
def test_shared_function_hysteresis_and_strict_boundaries(monkeypatch, on, normal_W, expected_run):
    monkeypatch.setattr('control.heat_pump_power.fixed_spread_heat_pump',
                        lambda *args: SimpleNamespace(heat_W=5000))
    c = configuration()
    p = build_simulation_params(c)
    state = initial_state(c, 21)
    state.storage = StorageState(45)
    state.heat_pump = OperatingState(on, 1000)
    # Outside = room means no building loss. Target is arbitrary, independent
    # of the configured heating curve; a different strategy can supply it.
    target = 45 + normal_W / 2000
    request = heat_pump_power_request(target, state, SimulationInput(21), p, 2, 2000)
    assert request.requested_heat_W == pytest.approx(normal_W)
    assert request.shutdown_check_heat_W == pytest.approx(normal_W + 4000)
    assert (request.power_fraction >= 0) == expected_run


@pytest.mark.parametrize('strategy', ['heating_curve', 'heating_curve_supply'])
@pytest.mark.parametrize('hysteresis', [-1, float('nan'), float('inf')])
def test_invalid_hysteresis_rejected(strategy, hysteresis):
    c = configuration()
    c['control'].setdefault(strategy, {})['temperature_hysteresis_K'] = hysteresis
    with pytest.raises(ValueError, match='temperature_hysteresis_K'):
        build_simulation_params(c)
    p = build_simulation_params(configuration())
    state = initial_state(configuration(), 1)
    with pytest.raises(ValueError, match='temperature_hysteresis_K'):
        heat_pump_power_request(45, state, SimulationInput(1), p, hysteresis, 2000)


def test_hysteresis_is_configured_per_strategy():
    c = configuration()
    c['control']['heating_curve']['temperature_hysteresis_K'] = 12
    c['control'].setdefault('heating_curve_supply', {})['temperature_hysteresis_K'] = 7
    p = build_simulation_params(c)
    assert p.heating_curve_temperature_hysteresis_K == 12
    assert p.heating_supply_curve_temperature_hysteresis_K == 7


def test_actual_minimum_accounts_for_pump_bounds_and_sensor_selection():
    c = configuration()
    c['storage']['control_sensor_layer'] = 3
    c['heat_pump']['min_volume_flow_L_h'] = 3000
    p = build_simulation_params(c)
    state = initial_state(c, 1)
    state.storage = StorageState(10, (60, 55, 50, 48, 45, 43, 40, 38, 35, 30))
    result = heat_pump_power_request(45, state, SimulationInput(1), p, 2, 2000)
    minimum = fixed_spread_heat_pump(30, 1, 0, p.heat_pump)
    assert minimum.supply_C < 35  # Minimum circulation lowers the lift.
    assert result.minimum_heat_W == minimum.heat_W
    assert result.temperature_correction_W == -10000
    assert result.requested_heat_W == 0


def test_machine_latches_at_minimum_then_stops_and_does_not_restart_below_minimum():
    c = configuration()
    c['control']['method'] = 'heating_curve'
    c['control']['heating_curve']['temperature_hysteresis_K'] = 4
    c['control']['interval_minutes'] = c['simulation']['dt_minutes']
    p = build_simulation_params(c)
    state = initial_state(c, 1)
    state.heat_pump = OperatingState(True, 1000)
    # Loss ~7.7 kW, sensor 49 C -> normal demand 0; +4 K check = loss (above minimum).
    state.storage = StorageState(49, (49,) * 9 + (30,))
    request = heat_pump_power_request(45, state, SimulationInput(1), p, 4, 2000)
    assert 0 < request.minimum_heat_W <= request.shutdown_check_heat_W
    held = simulate_step(state, SimulationInput(1), p)
    assert held.state.heat_pump.is_on
    assert held.heat_pump_requested_thermal_power_W == 0
    assert held.thermal_power_W > 0
    # Substep physics follows manufacturer minimum at the changing return.
    assert held.heat_pump_commanded_power_fraction == 0
    hot = replace(held.state, storage=StorageState(55, (55,) * 9 + (30,)))
    stopped = simulate_step(hot, SimulationInput(1), p)
    assert not stopped.state.heat_pump.is_on
    assert stopped.thermal_power_W == 0
    waiting = replace(stopped.state, storage=state.storage)
    assert not simulate_step(waiting, SimulationInput(1), p).state.heat_pump.is_on


def test_zero_hysteresis_and_infeasible_operating_point():
    c = configuration()
    p = build_simulation_params(c)
    state = initial_state(c, 1)
    state.heat_pump = OperatingState(True, 1000)
    state.storage = StorageState(55, (55,) * 9 + (30,))
    request = heat_pump_power_request(45, state, SimulationInput(1), p, 0, 2000)
    assert request.shutdown_check_heat_W == request.requested_heat_W
    assert request.power_fraction == -1
    state.storage = StorageState(p.heat_pump.sink_temp_max_C)
    assert heat_pump_power_request(100, state, SimulationInput(1), p, 0, 2000).power_fraction == -1


def test_hysteresis_minimum_hold_uses_achievable_minimum_not_nominal_supply(monkeypatch):
    monkeypatch.setattr('control.heat_pump_power.fixed_spread_heat_pump',
                        lambda *args: SimpleNamespace(heat_W=10000))
    c = configuration()
    p = build_simulation_params(c)
    state = initial_state(c, 1)
    state.storage = StorageState(45)
    state.heat_pump = OperatingState(True, 1000)
    request = heat_pump_power_request(45, state, SimulationInput(1), p, 2, 2000)
    assert request.requested_heat_W == pytest.approx(LOAD_W)
    assert request.shutdown_check_heat_W == pytest.approx(LOAD_W + 4000)
    assert request.power_fraction == 0  # Exact minimum hold, including pump-bound effects.
