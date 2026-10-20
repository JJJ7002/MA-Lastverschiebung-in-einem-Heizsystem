"""WP pump bounds, achieved supply and conservative heat transfer."""
from datetime import datetime
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support import load_test_config
from models.heat_pump import get_performance
from models.hydraulics import fixed_spread_heat_pump
from simulation.engine import build_simulation_params, initial_state, run_simulation, build_result_rows
from simulation.step import SimulationInput, simulate_step
from weather.loader import WeatherPoint


def configuration():
    c = load_test_config()
    c['simulation'].update(start_time=None, end_time=None)
    c['control']['night_setback']['enabled'] = False
    c['heat_pump'].update(min_on_time_min=0, min_off_time_min=0)
    return c












@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), 5000])
def test_invalid_minimum_or_reversed_bounds_rejected(value):
    c = configuration()
    c['heat_pump']['min_volume_flow_L_h'] = value
    with pytest.raises(ValueError, match='min_volume_flow_L_h'):
        build_simulation_params(c)


def test_minimum_flow_coupled_steps_conserve_tank_and_building_energy():
    c = configuration()
    c['control']['method'] = 'heating_curve'
    c['control']['heating_curve'] = {'slope': 0, 'offset_C': 60}
    c['storage']['T_storage_init_C'] = 25
    c['heat_pump']['min_volume_flow_L_h'] = 3000
    p = build_simulation_params(c)
    state = initial_state(c, 0)
    for _ in range(15):
        previous = state
        result = simulate_step(state, SimulationInput(0), p, .4)
        state = result.state
        assert result.wp_active_volume_flow_L_h == pytest.approx(3000)
        assert result.wp_supply_temperature_C < result.wp_supply_target_temperature_C
        tank_delta = p.storage.model.energy_Wh(state.storage.node_temperatures_C) - p.storage.model.energy_Wh(previous.storage.node_temperatures_C)
        assert tank_delta == pytest.approx((result.thermal_power_W - result.heating_power_to_building_W
                                           - result.storage_loss_W) * p.dt_hours, abs=1e-7)
        building_delta = p.building.energy_Wh(state.building) - p.building.energy_Wh(previous.building)
        assert building_delta == pytest.approx((result.heating_power_to_building_W - result.building_heat_loss_W)
                                           * p.dt_hours, abs=1e-7)


def test_csv_separates_active_flow_from_time_average_for_partial_runtime():
    c = configuration()
    c['control']['method'] = 'heating_curve'
    c['control']['heating_curve'] = {'slope': 0, 'offset_C': 43}
    c['building']['UA_W_per_K'] = 0
    c['storage'].update(V_storage_L=20, node_count=1, T_storage_init_C=63)
    c['control']['heating_curve'] = {'slope_factor': 1, 'offset_K': 43, 'temperature_gain_W_per_K': 100000}
    result = run_simulation(c, weather_points=[WeatherPoint(datetime(2025, 1, 1, 12), 0)])
    row = build_result_rows(result)[0]
    assert 0 < row['heat_pump_runtime_fraction'] < 1
    assert 1000 <= row['wp_active_volume_flow_L_h'] <= c['heat_pump']['max_volume_flow_L_h'] + 1e-8
    assert row['wp_volume_flow_L_h'] == pytest.approx(
        row['wp_active_volume_flow_L_h'] * row['heat_pump_runtime_fraction'])


@pytest.mark.parametrize('method', ['heating_curve', 'cop_optimized',
                                   'pause_load_shifting', 'reinforcement_learning'])
def test_all_controllers_honor_minimum_flow_and_evaluate_achieved_supply(method):
    from unittest.mock import patch
    c = configuration()
    c['control']['method'] = method
    c['storage']['T_storage_init_C'] = 25
    c['building']['T_room_init_C'] = 19
    instant = datetime(2025, 1, 1, 12)
    p = build_simulation_params(c).heat_pump
    flows = []
    def capture(*args):
        flow = fixed_spread_heat_pump(*args)
        flows.append(flow)
        return flow
    with patch('simulation.step.fixed_spread_heat_pump', side_effect=capture), \
         patch('control.rl_power.RLController') as controller:
        controller.return_value.predict.return_value = .4
        result = run_simulation(c, weather_points=[WeatherPoint(instant, -7)],
                                daily_temperatures={instant.date(): -7})
    active = [flow for flow in flows if flow.heat_W > 0]
    assert active
    assert all(1000 - 1e-8 <= flow.mass_kg_s * 3600 <= p.max_volume_flow_L_h + 1e-8 for flow in active)
    assert all(flow.supply_C <= p.sink_temp_max_C for flow in active)
    for flow in active:
        assert (flow.cop, flow.heat_W, flow.electricity_W) == pytest.approx(
            get_performance(-7, flow.supply_C, p, flow.fraction), abs=1e-6)
