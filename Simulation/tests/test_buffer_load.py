from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support import load_test_config
from control.heating_curve import heating_curve_power_request
from models.storage import StorageState
from models.hydraulics import fixed_spread_heat_pump
from models.heat_pump import available_power_fractions, get_performance, heat_to_power_command
from simulation.engine import build_simulation_params, initial_state, run_simulation, build_result_rows
from simulation.step import SimulationInput, simulate_step
from weather.loader import WeatherPoint


def _stationary_load_W():
    from models.building import BuildingParams
    building = dict(load_test_config()['building'], UA_W_per_K=400)
    return BuildingParams.from_config(building).stationary_UA_W_per_K * (21 - 1)


LOAD_W = _stationary_load_W()  # steady-state building loss at 21 C room, 1 C outside


def configuration():
    c = load_test_config()
    c['control']['method'] = 'heating_curve'  # These tests cover the buffer-curve controller.
    c['control']['night_setback']['enabled'] = False
    c['heat_pump'].update(min_on_time_min=0, min_off_time_min=0)
    c['control']['heating_curve'] = {'slope': 0, 'offset_C': 45}
    c['building'].update(T_room_init_C=21, T_set_C=21, UA_W_per_K=400)
    from models.radiator import RadiatorMap
    radiator = RadiatorMap(volume_flow_L_h=c['heating']['volume_flow_L_h'], **c['heating']['radiator'])
    c['control']['heating_curve'] = {'slope_factor': 1, 'offset_K': 45-21-radiator.temperatures(LOAD_W)[0], 'temperature_gain_W_per_K': 2000}
    return c


@pytest.mark.parametrize('sensor,error', [(43, 2), (45, 0), (47, -2), (50, -5)])
def test_signed_formula_uses_current_loss_and_second_bottom_sensor(sensor, error):
    c = configuration()
    p = build_simulation_params(c)
    state = initial_state(c, 1)
    state.storage = StorageState(60, (60,) * 8 + (sensor, 25))
    request, target = heating_curve_power_request(state, SimulationInput(1), p)
    assert target == pytest.approx(45)
    assert request.requested_heat_W == pytest.approx(max(0, LOAD_W + 2000 * error))


@pytest.mark.parametrize('gain', [0, 500, 2000])
def test_gain_and_signed_building_loss_are_configurable(gain):
    c = configuration()
    c['control']['heating_curve']['temperature_gain_W_per_K'] = gain
    p = build_simulation_params(c)
    state = initial_state(c, 23)
    state.storage = StorageState(43)
    request, target = heating_curve_power_request(state, SimulationInput(23), p)
    assert request.requested_heat_W == pytest.approx(max(0, -800 + (target-43) * gain))


@pytest.mark.parametrize('gain', [-1, float('nan'), float('inf')])
def test_invalid_gain_rejected(gain):
    c = configuration()
    c['control']['heating_curve']['temperature_gain_W_per_K'] = gain
    with pytest.raises(ValueError, match='temperature_gain_W_per_K'):
        build_simulation_params(c)


def test_no_hysteresis_settings_needed_and_setback_applies_to_single_curve_target():
    c = configuration()
    c['control'].pop('hysteresis', None)
    c['control']['night_setback'].update(enabled=True, start='22:00', end='06:00', offset_K=3,
                                         warmup_minutes_pro_K=0)
    p = build_simulation_params(c)
    state = initial_state(c, 1)
    state.storage = StorageState(43)
    request, target = heating_curve_power_request(state, SimulationInput(1, datetime(2025, 1, 1, 23)), p)
    expected = p.operating.heating_curve.target_temperature_C(1, c['building']['T_set_C'] - 3)
    assert target == pytest.approx(expected)
    stationary_loss = p.building.stationary_UA_W_per_K * (19 - 1)
    assert request.requested_heat_W == pytest.approx(max(0, stationary_loss+2000*(expected-43)))


@pytest.mark.parametrize('return_C', [25, 35, 45])
def test_watts_are_converted_to_modulation_at_current_return(return_C):
    p = build_simulation_params(configuration()).heat_pump
    lift = p.supply_spread_K
    command = heat_to_power_command(10000, 0, return_C + lift, p)
    flow = fixed_spread_heat_pump(return_C, 0, command, p)
    # The command is the smallest manufacturer map level that covers the request.
    assert flow.heat_W >= 10000
    lower = [level for level in available_power_fractions(p) if level < command]
    if lower:
        assert fixed_spread_heat_pump(return_C, 0, lower[-1], p).heat_W < 10000
    assert flow.supply_C == return_C + lift
    assert flow.electricity_W == pytest.approx(flow.heat_W / flow.cop)
    assert flow.cop == pytest.approx(get_performance(0, flow.supply_C, p, flow.fraction)[0])


def test_modulation_is_kept_at_both_pump_bounds():
    p = build_simulation_params(configuration()).heat_pump
    for min_flow, max_flow in [(3000, 4600), (1000, 1000)]:
        bounded = replace(p, min_volume_flow_L_h=min_flow, max_volume_flow_L_h=max_flow)
        flow = fixed_spread_heat_pump(25, 0, .8, bounded)
        assert flow.fraction == pytest.approx(.8)
        assert flow.limited
        assert flow.cop == pytest.approx(get_performance(0, flow.supply_C, bounded, flow.fraction)[0])


def test_modulation_is_held_between_decisions_while_watts_follow_operating_point():
    c = configuration()
    c['storage']['T_storage_init_C'] = 45
    p = build_simulation_params(c)
    state = initial_state(c, 1)
    outputs = []
    for _ in range(p.control_steps):
        result = simulate_step(state, SimulationInput(1), p)
        outputs.append(result)
        state = result.state
    assert all(r.heat_pump_requested_thermal_power_W == pytest.approx(LOAD_W) for r in outputs)
    assert len({r.heat_pump_commanded_power_fraction for r in outputs}) == 1
    assert len({round(r.thermal_power_W, 5) for r in outputs}) > 1
    result = run_simulation(c, weather_points=[WeatherPoint(datetime(2025, 1, 1, 12), 1)])
    row = build_result_rows(result)[0]
    assert row['buffer_target_C'] == pytest.approx(45)
    assert row['heat_pump_requested_thermal_power_W'] == pytest.approx(LOAD_W)
    assert row['heat_pump_building_loss_feedforward_W'] == pytest.approx(LOAD_W)
    assert row['heat_pump_buffer_temperature_correction_W'] == pytest.approx(0, abs=1e-6)


def test_minimum_runtime_and_blocking_schedule_are_respected():
    c = configuration()
    c['heat_pump']['min_on_time_min'] = 30
    p = build_simulation_params(c)
    state = initial_state(c, 1)
    from control.operating_constraints import OperatingState
    state.heat_pump = OperatingState(True, 0)
    state.storage = StorageState(55, (55,) * 9 + (30,))
    result = simulate_step(state, SimulationInput(1), p)
    assert result.heat_pump_requested_thermal_power_W == 0
    assert result.thermal_power_W > 0  # Manufacturer minimum until equipment runtime expires.
    c['control']['schedule'] = {'enabled': True, 'windows': [{'start': '11:00', 'end': '13:00', 'mode': 'force_off'}]}
    result = simulate_step(state, SimulationInput(1, datetime(2025, 1, 1, 12)), build_simulation_params(c))
    assert result.thermal_power_W == 0
