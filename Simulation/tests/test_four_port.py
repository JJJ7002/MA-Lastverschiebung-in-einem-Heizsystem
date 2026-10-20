"""Physical balances and limits for the stratified optimization plant."""
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support import load_test_config
from control.operating_constraints import OperatingState
from models.hydraulics import ideal_heating, fixed_spread_heat_pump, four_ports
from models.storage import StorageState
from simulation.engine import build_simulation_params, initial_state, run_simulation, build_result_rows
from simulation.step import SimulationInput, simulate_step
from weather.loader import WeatherPoint

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    c = load_test_config()
    c['simulation'].update(start_time=None, end_time=None)
    c['control']['night_setback']['enabled'] = False
    c['heat_pump'].update(min_on_time_min=0, min_off_time_min=0)
    return c


def test_sbp_four_ports_use_requested_layers_and_conserve_heat_and_mass():
    params = build_simulation_params(configuration())
    temps = np.linspace(55, 25, params.storage.node_count)
    hp = fixed_spread_heat_pump(25, 0, .6, params.heat_pump)
    heating = ideal_heating(55, 21, 8000, 350, 1500)
    ports = four_ports(len(temps), hp, heating)
    assert [p.node for p in ports] == [1, len(temps) - 1, 0, len(temps) - 2]
    assert sum(p.mass_flow_kg_s for p in ports) == pytest.approx(0)
    tank = params.storage.model
    dt = min(10, tank.max_step_seconds(temps, ports, 21))
    result = tank.step(temps, ports, 21, dt)
    assert result.port_heat_W == pytest.approx(hp.heat_W - heating.heat_W)
    assert result.energy_change_Wh == pytest.approx((hp.heat_W - heating.heat_W - result.loss_W) * dt / 3600)
    assert abs(result.balance_error_Wh) < 1e-8


def test_all_nodes_lose_heat_and_conduction_preserves_total_energy():
    params = build_simulation_params(configuration()).storage
    tank = replace(params, ua_W_per_K=3).model
    old = np.linspace(60, 30, params.node_count)
    result = tank.step(old, (), 20, 10)
    assert result.loss_W == pytest.approx(3 * (old.mean() - 20))
    assert result.energy_change_Wh == pytest.approx(-result.loss_W * 10 / 3600)
    unstable = old.copy()
    unstable[3], unstable[4] = 35, 50
    mixed = tank.mix_inversions(unstable)
    assert np.all(mixed[:-1] >= mixed[1:])
    assert tank.energy_Wh(mixed) == pytest.approx(tank.energy_Wh(unstable))


def test_coupled_step_conserves_energy_and_does_not_mutate_input():
    c = configuration()
    c['control']['method'] = 'heating_curve'
    c['storage']['initial_node_temperatures_C'] = list(np.linspace(55, 25, 10))
    c['storage']['UA_storage_W_per_K'] = 3
    params = build_simulation_params(c)
    state = initial_state(c, 0)
    original = state.storage.node_temperatures_C
    result = simulate_step(state, SimulationInput(0, datetime(2025, 1, 1, 12)), params, .6)
    tank = params.storage.model
    delta = tank.energy_Wh(result.state.storage.node_temperatures_C) - tank.energy_Wh(original)
    assert delta == pytest.approx((result.thermal_power_W - result.heating_power_to_building_W
                                  - result.storage_loss_W) * params.dt_hours, abs=1e-7)
    building_delta = params.building.energy_Wh(result.state.building) - params.building.energy_Wh(state.building)
    assert building_delta == pytest.approx((result.heating_power_to_building_W - result.building_heat_loss_W)
                                      * params.dt_hours, abs=1e-7)
    assert state.storage.node_temperatures_C == original
    assert result.wp_supply_temperature_C <= params.heat_pump.sink_temp_max_C + 1e-8
    assert result.wp_active_volume_flow_L_h >= params.heat_pump.min_volume_flow_L_h - 1e-8
    assert result.wp_return_temperature_C < state.storage.temperature_C


def test_results_export_every_node_and_both_hydraulic_circuits():
    c = configuration()
    result = run_simulation(c, weather_points=[WeatherPoint(datetime(2025, 1, 1, 12), 0)])
    row = build_result_rows(result)[0]
    assert row['storage_temperature_C'] == row['storage_control_temperature_C'] == row['storage_node_9_temperature_C']
    assert row['storage_mean_temperature_C'] == pytest.approx(np.mean(result.storage_node_temperatures_C[0]))
    assert row['storage_top_temperature_C'] > row['storage_temperature_C']
    assert row['storage_top_temperature_C'] == row['storage_node_1_temperature_C']
    assert row['storage_bottom_temperature_C'] == row['storage_node_10_temperature_C']
    assert 'wp_volume_flow_L_h' in row and 'heating_mixer_fraction' in row


def test_heating_curve_keeps_charging_with_hot_top_and_cold_control_layer():
    c = configuration()
    c['control']['method'] = 'heating_curve'
    c['control']['heating_curve'] = {'slope': 0, 'offset_C': 45}
    c['storage']['initial_node_temperatures_C'] = [50] + [30] * 9
    params = build_simulation_params(c)
    state = initial_state(c, 0)
    state.heat_pump = OperatingState(True, 60)
    assert state.storage.temperature_C == state.storage.control_C == 30
    assert state.storage.top_C == 50
    assert state.storage.mean_temperature_C == 32
    result = simulate_step(state, SimulationInput(0), params)
    assert result.heat_pump_commanded_power_fraction == 1
    assert result.thermal_power_W > 0


@pytest.mark.parametrize('method', ['heating_curve', 'cop_optimized', 'reinforcement_learning'])
def test_temperature_ceiling_uses_control_layer_for_all_power_controllers(method):
    c = configuration()
    c['control']['method'] = method
    c['storage']['initial_node_temperatures_C'] = [65] * 9 + [30]
    instant = datetime(2025, 1, 1, 12)
    params = build_simulation_params(c)
    if method == 'cop_optimized':
        from control.cop_optimized import CopOptimizedParams
        params.cop_control = CopOptimizedParams({instant.date(): 3})
    state = initial_state(c, 0)
    assert state.storage.bottom_C == 30
    assert state.storage.control_C == 65
    result = simulate_step(state, SimulationInput(0, instant), params,
                           None if method == 'cop_optimized' else 1)
    assert result.heat_pump_commanded_power_fraction == -1
    assert result.thermal_power_W == 0


def test_profile_loss_uses_individual_layer_weights():
    params = replace(build_simulation_params(configuration()).storage, ua_W_per_K=3)
    nodes = (60,) + (40,) * 8 + (25,)
    state = StorageState(60, nodes)
    expected = sum(ua * (temperature - 21) for ua, temperature in zip(params.model.ua_W_K, nodes))
    assert params.heat_loss_W(state, 21) == pytest.approx(expected)
    assert expected != pytest.approx(params.ua_W_per_K * (state.mean_temperature_C - 21))


def test_cop_reference_uses_the_same_supply_target_as_current_cop():
    c = configuration()
    c['control']['method'] = 'cop_optimized'
    c['building']['UA_W_per_K'] = 0
    # Return exactly at the daily buffer-curve target: reference and current COP
    # are then both evaluated at target + lift and must coincide.
    curve = build_simulation_params(c).operating.heating_curve
    c['storage']['T_storage_init_C'] = curve.target_temperature_C(0)
    instant = datetime(2025, 1, 1, 12)
    result = run_simulation(c, weather_points=[WeatherPoint(instant, 0)],
                            daily_temperatures={instant.date(): 0})
    assert result.cop_advantage[0] == pytest.approx(0)


def test_start_and_short_runtime_are_counted_even_when_off_at_step_end():
    from analysis.metrics import calculate_metrics
    from analysis.scoring import ScoringParams, calculate_score
    c = configuration()
    c['control']['method'] = 'heating_curve'
    c['control']['heating_curve'] = {'slope': 0, 'offset_C': 43}
    c['building']['UA_W_per_K'] = 0
    c['storage'].update(V_storage_L=20, node_count=1, T_storage_init_C=63)
    c['control']['heating_curve'] = {'slope_factor': 1, 'offset_K': 43, 'temperature_gain_W_per_K': 100000}
    result = run_simulation(c, weather_points=[WeatherPoint(datetime(2025, 1, 1, 12), 0)])
    assert result.heat_pump_on == [False]
    assert result.thermal_power_W[0] > 0
    assert result.heat_pump_starts == [1]
    share = result.heat_pump_runtime_fractions[0]
    assert 0 < share < 1
    dt = c['simulation']['dt_minutes'] / 60
    metrics = calculate_metrics(result, .1, dt, 21)
    assert metrics.heat_pump_start_count == 1
    assert metrics.heat_pump_on_share == pytest.approx(share)
    scoring = ScoringParams.from_config(c['scoring'])
    score = calculate_score(result, metrics, dt, scoring)
    assert score.cycle_cost == scoring.cost_per_start
    incremental = scoring.transition_score(datetime(2025, 1, 1, 12), dt,
        result.electric_power_W[0], False, False, 21, 20.9, start_count=1)
    assert incremental.cycle_cost == score.cycle_cost
