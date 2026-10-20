"""SBP 400 E data, port positions and second-bottom control sensor."""
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from math import pi
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support import load_test_config
from control.operating_constraints import OperatingState
from weather.loader import WeatherPoint
from models.hydraulics import HeatPumpFlow, ideal_heating, four_ports
from models.storage import StorageState
from models.stratified_tank import CP_J_KG_K
from simulation.engine import build_simulation_params, initial_state, run_simulation, build_result_rows
from simulation.step import SimulationInput, simulate_step


def configuration():
    c = load_test_config()
    c["building"].update(T_mass_init_C=21.0)
    c['control']['night_setback']['enabled'] = False
    c['heat_pump'].update(min_on_time_min=0, min_off_time_min=0)
    return c


def test_manufacturer_volume_capacity_and_reference_standby_loss():
    c = configuration()
    c['storage']['model'] = 'SBP 400 E'
    p = build_simulation_params(c).storage
    assert p.volume_L == 415
    assert CP_J_KG_K == 4182.0
    assert p.heat_capacity_Wh_per_K == pytest.approx(415 * 4182 / 3600)
    state = StorageState(65, (65,) * p.node_count)
    assert p.heat_loss_W(state, 20) * 24 == pytest.approx(1600)
    assert p.model.energy_Wh([1] * p.node_count) == pytest.approx(p.heat_capacity_Wh_per_K)
    assert pi * p.diameter_m**2 / 4 * p.water_height_m * 1000 == pytest.approx(415, abs=1e-4)


@pytest.mark.parametrize(('model', 'volume', 'standby_kwh', 'diameter_m'), [
    ('SBP 200 E', 207, 1.10, 0.450),
    ('SBP 400 E', 415, 1.60, 0.600),
    ('SBP 700 E', 720, 2.20, 0.749),
])
def test_named_sbp_e_model_loads_volume_standby_loss_and_vessel_diameter(model, volume, standby_kwh, diameter_m):
    c = configuration()
    c['storage']['model'] = model
    p = build_simulation_params(c).storage
    assert p.volume_L == volume
    assert p.diameter_m == diameter_m
    assert p.heat_loss_W(StorageState(65), 20) * 24 / 1000 == pytest.approx(standby_kwh)
    from simulation.engine import _load_storage_profile
    profile = _load_storage_profile(c['storage'])
    expected_ua = standby_kwh * 1000 / (24 * (65 - 20))
    expected_height = volume / 1000 / (pi * profile['diameter_m']**2 / 4)
    expected_area = pi * profile['diameter_m'] * expected_height + pi * profile['diameter_m']**2 / 2
    assert profile['ua_W_per_K'] == pytest.approx(expected_ua)
    assert profile['equivalent_cylinder_area_m2'] == pytest.approx(expected_area)
    import yaml
    models = yaml.safe_load((Path(__file__).resolve().parents[1] / 'data/sbp_e_models.yaml').read_text(encoding='utf-8'))
    assert 'ua_W_per_K' not in models[model]


def test_unknown_storage_model_has_actionable_error():
    c = configuration()
    c['storage']['model'] = 'SBP 999 E'
    with pytest.raises(ValueError, match='Unknown storage.model.*SBP 200 E.*SBP 700 E'):
        build_simulation_params(c)


@pytest.mark.parametrize('model', ['SBP 1000 E', 'SBP 1500 E'])
def test_large_sbp_models_are_not_selectable(model):
    c = configuration()
    c['storage']['model'] = model
    with pytest.raises(ValueError, match='Unknown storage.model'):
        build_simulation_params(c)


@pytest.mark.parametrize('node_count', [1, 2, 4, 10, 20])
def test_ports_and_control_sensor_follow_relative_layer_positions(node_count):
    p = replace(build_simulation_params(configuration()).storage, node_count=node_count)
    hp = HeatPumpFlow(50, 20, .1, .1 * CP_J_KG_K * 30, 1000, 3, .5)
    heating = ideal_heating(60, 21, 5000, 350, 1500, supply_target_C=40)
    ports = four_ports(node_count, hp, heating)
    assert [flow.node for flow in ports] == [p.wp_supply_node, p.wp_return_node,
                                            p.heating_supply_node, p.heating_return_node]
    assert p.control_sensor_node == max(0, node_count - 2)
    assert sum(flow.mass_flow_kg_s for flow in ports) == pytest.approx(0)
    temperatures = tuple(np.linspace(60, 20, node_count))
    state = StorageState(0, temperatures)
    assert state.temperature_C == state.control_C == temperatures[p.control_sensor_node]
    assert state.top_C == temperatures[0] and state.bottom_C == temperatures[-1]


def test_wp_enters_second_layer_without_directly_heating_top_layer():
    p = replace(build_simulation_params(configuration()).storage, ua_W_per_K=0)
    temperatures = np.asarray([60, 40] + [35] * 7 + [20], dtype=float)
    hp = HeatPumpFlow(50, 20, .1, .1 * CP_J_KG_K * 30, 1000, 3, .5)
    heating = ideal_heating(60, 21, 0, 350, 1500)
    result = p.model.step(temperatures, four_ports(10, hp, heating), 21, 1)
    assert result.temperatures_C[0] == pytest.approx(temperatures[0], abs=1e-3)
    assert result.temperatures_C[1] > temperatures[1]
    assert result.port_heat_W == pytest.approx(hp.heat_W)
    assert result.energy_change_Wh == pytest.approx(hp.heat_W / 3600)
    assert abs(result.balance_error_Wh) < 1e-8


def test_heating_return_enters_second_bottom_and_leaves_bottom_untouched():
    p = replace(build_simulation_params(configuration()).storage, ua_W_per_K=0)
    temperatures = np.asarray([60] + [55] * 7 + [50, 20], dtype=float)
    hp = HeatPumpFlow(None, 20, 0, 0, 0, 0, 0)
    heating = ideal_heating(60, 21, 5000, 350, 1500, supply_target_C=40)
    result = p.model.step(temperatures, four_ports(10, hp, heating), 21, 1)
    assert result.temperatures_C[-1] == pytest.approx(temperatures[-1], abs=1e-3)
    assert result.temperatures_C[-2] < temperatures[-2]
    assert result.port_heat_W == pytest.approx(-heating.heat_W)
    assert result.energy_change_Wh == pytest.approx(-heating.heat_W / 3600)
    assert abs(result.balance_error_Wh) < 1e-8


@pytest.mark.parametrize('sensor_C', [44, 46, 48])
def test_wp_curve_uses_second_bottom_instead_of_hot_top_or_cold_bottom(sensor_C):
    c = configuration()
    c['control']['method'] = 'heating_curve'
    c['control']['heating_curve'] = {'slope': 0, 'offset_C': 45}
    c['storage']['initial_node_temperatures_C'] = [60] * 8 + [sensor_C, 20]
    p = build_simulation_params(c)
    state = initial_state(c, 0)
    state.heat_pump = OperatingState(True, 60)
    result = simulate_step(state, SimulationInput(0), p)
    from control.heating_curve import heating_curve_power_request
    expected = heating_curve_power_request(state, SimulationInput(0), p)[0].power_fraction
    assert result.heat_pump_commanded_power_fraction == expected
    assert (result.thermal_power_W > 0) == (expected >= 0)


@pytest.mark.parametrize('reference', ['supply', 'return'])
def test_coupled_balances_and_exported_sensor_for_both_circuit_controls(reference):
    c = configuration()
    c['control']['heating_circuit_curve'] = {'slope': -1.4, 'reference': reference, 'offset_C': 40}
    c['storage']['initial_node_temperatures_C'] = list(np.linspace(60, 25, 10))
    p = build_simulation_params(c)
    start = initial_state(c, 0)
    instant = datetime(2025, 1, 1, 12)
    result = run_simulation(c, weather_points=[WeatherPoint(instant, 0)])
    row = build_result_rows(result)[0]
    assert row['storage_control_sensor_layer'] == 9
    assert row['storage_temperature_C'] == row['storage_control_temperature_C'] == row['storage_node_9_temperature_C']
    assert [row[key] for key in ('wp_supply_storage_layer', 'wp_return_storage_layer',
                                'heating_supply_storage_layer', 'heating_return_storage_layer')] == [2, 10, 1, 9]
    delta = p.storage.model.energy_Wh(result.storage_node_temperatures_C[0]) - p.storage.model.energy_Wh(start.storage.node_temperatures_C)
    assert delta == pytest.approx((row['thermal_power_W'] - row['heating_power_to_building_W']
                                  - row['storage_loss_W']) * p.dt_hours, abs=1e-7)
