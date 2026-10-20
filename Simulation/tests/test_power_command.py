"""Common command contract and operating-point tracking without controller updates."""

import pytest

from test_buffer_load import configuration
from models.heat_pump import get_performance, heat_to_power_command
from models.hydraulics import fixed_spread_heat_pump
from simulation.engine import build_simulation_params, initial_state
from simulation.step import SimulationInput, simulate_step
from control.rl_power import apply_power_request, power_levels_from_config


def c_map_levels(config):
    return build_simulation_params(config).heat_pump.performance_map.power_fractions


@pytest.mark.parametrize('source,supply', [(-7, 35), (7, 45), (20, 55)])
def test_command_endpoints_and_midpoint_use_current_minimum_and_maximum(source, supply):
    p = build_simulation_params(configuration()).heat_pump
    minimum = p.performance_map.curves(source, supply)[0][0]
    maximum = p.performance_map.curves(source, supply)[-1][0]
    assert get_performance(source, supply, p, -1) == (0, 0, 0)
    for command in (0, .25, .5, 1):
        cop, heat, electric = get_performance(source, supply, p, command)
        speed_index = round(command * (len(p.performance_map.speeds) - 1))
        assert heat == pytest.approx(p.performance_map.curves(source, supply)[speed_index][0])
        assert heat_to_power_command(heat, source, supply, p) == pytest.approx(command)
        assert electric * cop == pytest.approx(heat)


@pytest.mark.parametrize('command', [-2, -.5, 1.1, float('nan'), float('inf')])
def test_invalid_commands_are_rejected_at_all_plant_boundaries(command):
    c = configuration()
    p = build_simulation_params(c)
    calls = [lambda: get_performance(0, 35, p.heat_pump, command),
             lambda: fixed_spread_heat_pump(30, 0, command, p.heat_pump),
             lambda: simulate_step(initial_state(c, 0), SimulationInput(0), p, command),
             lambda: apply_power_request(command, 35, initial_state(c, 0).heat_pump,
                 p.operating, p.heat_pump, p.step_minutes, None)]
    for call in calls:
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize('command', [0, .5, 1])
def test_held_command_tracks_changing_weather_between_decisions(command):
    c = configuration()
    c['storage'].update(V_storage_L=1e12, T_storage_init_C=30)
    c['heat_pump'].update(min_volume_flow_L_h=0, max_volume_flow_L_h=None)
    p = build_simulation_params(c)
    first = simulate_step(initial_state(c, -15), SimulationInput(-15), p, command)
    second = simulate_step(first.state, SimulationInput(7), p)
    assert first.state.commanded_power_fraction == second.state.commanded_power_fraction == command
    assert first.thermal_power_W != pytest.approx(second.thermal_power_W)
    for result, outside in ((first, -15), (second, 7)):
        supply = 30 + p.heat_pump.supply_spread_K  # Constant return + configured spread.
        assert result.thermal_power_W == pytest.approx(get_performance(outside, supply, p.heat_pump, command)[1])
        assert result.state.heat_pump.is_on  # Includes command 0 = minimum.


def test_rl_has_distinct_off_and_minimum_actions_without_artificial_floor():
    c = configuration()
    assert power_levels_from_config(c)[:2] == (-1, 0)
    assert power_levels_from_config(c)[1:] == c_map_levels(c)  # Off + every map level (0 = minimum)
    p = build_simulation_params(c)
    state, command = apply_power_request(0, 35, initial_state(c, 0).heat_pump,
                                        p.operating, p.heat_pump, p.step_minutes, None)
    assert state.is_on and command == 0
