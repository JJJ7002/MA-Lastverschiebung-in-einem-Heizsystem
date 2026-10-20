"""Fixed WP spread, clipped pump flow, achieved supply and manufacturer map."""
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support import load_test_config
from models.heat_pump import get_performance
from models.hydraulics import fixed_spread_heat_pump
from models.stratified_tank import CP_J_KG_K
from simulation.engine import build_simulation_params, initial_state, run_simulation, build_result_rows
from simulation.step import simulate_step, SimulationInput
from weather.loader import WeatherPoint


def configuration():
    c = load_test_config()
    c['simulation'].update(start_time=None, end_time=None)
    c['control']['night_setback']['enabled'] = False
    c['heat_pump'].update(min_on_time_min=0, min_off_time_min=0)
    return c


def check_energy_and_map(flow, outside, p):
    assert flow.heat_W == pytest.approx(flow.mass_kg_s * CP_J_KG_K * (flow.supply_C - flow.return_C), abs=1e-7)
    assert flow.cop * flow.electricity_W == pytest.approx(flow.heat_W)
    assert (flow.cop, flow.heat_W, flow.electricity_W) == pytest.approx(
        get_performance(outside, flow.supply_C, p, flow.fraction), abs=1e-6)


@pytest.mark.parametrize('spread', [3, 5, 10])
@pytest.mark.parametrize('return_C', [30, 45])
def test_configurable_spread_and_map_supply_are_independent_of_buffer_target(spread, return_C):
    p = replace(build_simulation_params(configuration()).heat_pump,
                supply_spread_K=spread, min_volume_flow_L_h=0, max_volume_flow_L_h=None)
    flow = fixed_spread_heat_pump(return_C, 7, .8, p)
    assert flow.supply_C == return_C + spread
    assert flow.mass_kg_s == pytest.approx(flow.heat_W / (CP_J_KG_K * spread))
    assert not flow.limited
    check_energy_and_map(flow, 7, p)


@pytest.mark.parametrize('command', [.001, .4, 1])
def test_minimum_flow_reduces_spread_and_uses_reached_supply_in_map(command):
    p = replace(build_simulation_params(configuration()).heat_pump, min_volume_flow_L_h=3000)
    flow = fixed_spread_heat_pump(25, 0, command, p)
    assert flow.mass_kg_s * 3600 == pytest.approx(3000)
    assert 25 < flow.supply_C < 30
    assert flow.limited
    check_energy_and_map(flow, 0, p)


def test_maximum_flow_increases_spread_without_discarding_requested_heat():
    p = replace(build_simulation_params(configuration()).heat_pump, max_volume_flow_L_h=1000)
    flow = fixed_spread_heat_pump(25, 0, 1, p)
    assert flow.mass_kg_s * 3600 == pytest.approx(1000)
    assert flow.supply_C > 30
    assert flow.limited
    assert flow.fraction == pytest.approx(1)
    check_energy_and_map(flow, 0, p)


def test_absolute_supply_ceiling_reduces_heat_or_stops_below_compressor_minimum():
    p = replace(build_simulation_params(configuration()).heat_pump, max_volume_flow_L_h=2000)
    flow = fixed_spread_heat_pump(61, 0, 1, p)
    assert flow.supply_C == pytest.approx(65)
    assert flow.mass_kg_s * 3600 <= 2000 + 1e-6
    assert flow.fraction < 1
    check_energy_and_map(flow, 0, p)
    stopped = fixed_spread_heat_pump(64, 0, 1, p)
    assert stopped.heat_W == stopped.mass_kg_s == stopped.electricity_W == 0


@pytest.mark.parametrize('command,return_C', [(-1, 25), (1, 65), (1, 66)])
def test_shutdown_has_no_pump_flow(command, return_C):
    p = build_simulation_params(configuration()).heat_pump
    flow = fixed_spread_heat_pump(return_C, 0, command, p)
    assert flow.heat_W == flow.electricity_W == flow.mass_kg_s == 0


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_invalid_spread_rejected(value):
    c = configuration()
    c['heat_pump']['supply_spread_K'] = value
    with pytest.raises(ValueError, match='supply_spread_K'):
        build_simulation_params(c)


def test_spread_updates_from_actual_return_between_control_decisions():
    c = configuration()
    c['control']['method'] = 'heating_curve'
    c['control']['heating_curve'] = {'slope': 0, 'offset_C': 60}
    c['storage']['T_storage_init_C'] = 30
    p = build_simulation_params(c)
    first = simulate_step(initial_state(c, 0), SimulationInput(0), p, .8)
    second = simulate_step(first.state, SimulationInput(0), p)
    assert second.state.commanded_power_fraction == first.state.commanded_power_fraction
    assert second.buffer_target_C == first.buffer_target_C
    assert second.wp_supply_target_temperature_C != first.wp_supply_target_temperature_C
    for result in (first, second):
        spread = p.heat_pump.supply_spread_K
        assert result.wp_supply_temperature_C - result.wp_return_temperature_C == pytest.approx(spread)
        assert result.wp_supply_target_temperature_C - result.wp_return_temperature_C == pytest.approx(spread)


def test_cop_normalizes_at_return_plus_spread_not_buffer_curve():
    c = configuration()
    c['storage']['T_storage_init_C'] = 30
    c['control']['heating_curve'] = {'slope': 0, 'offset_C': 55}
    c['control']['method'] = 'cop_optimized'
    instant = datetime(2025, 1, 1, 12)
    result = run_simulation(c, weather_points=[WeatherPoint(instant, 0)],
                            daily_temperatures={instant.date(): 0})
    p = build_simulation_params(c)
    row = build_result_rows(result)[0]
    expected_target = p.operating.heating_curve.target_temperature_C(0)
    assert row['buffer_target_C'] == pytest.approx(expected_target)
    spread = p.heat_pump.supply_spread_K
    assert row['potential_cop'] == pytest.approx(get_performance(0, 30 + spread, p.heat_pump)[0])
    assert row['reference_cop'] == pytest.approx(get_performance(0, min(65, expected_target + spread), p.heat_pump)[0])
    assert row['wp_supply_target_temperature_C'] - row['wp_return_temperature_C'] == pytest.approx(spread)
