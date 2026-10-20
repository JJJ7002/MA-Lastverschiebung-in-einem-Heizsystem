"""Selectable sensor shared by all controllers, exports and RL observations."""
from datetime import datetime
from pathlib import Path
import sys
from unittest.mock import patch, Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support import load_test_config
from control.heating_curve import heating_curve_power_request
from control.rl_power import RLController
from models.storage import StorageState
from models.hydraulics import fixed_spread_heat_pump
from simulation.engine import build_simulation_params, initial_state, run_simulation, build_result_rows
from simulation.step import SimulationInput, simulate_step
from weather.loader import WeatherPoint


def configuration(layer=None):
    c = load_test_config()
    c['storage']['control_sensor_layer'] = layer
    c['storage']['initial_node_temperatures_C'] = [float(t) for t in np.linspace(60, 20, 10)]
    c['control']['night_setback']['enabled'] = False
    c['simulation'].update(start_time=None, end_time=None, settling_hours=0)
    c['heat_pump'].update(min_on_time_min=0, min_off_time_min=0)
    return c


@pytest.mark.parametrize('layer', [None, 1, 5, 9, 10])
def test_initial_state_and_parameters_resolve_selected_sensor(layer):
    c = configuration(layer)
    p, state = build_simulation_params(c).storage, initial_state(c, 0).storage
    node = 8 if layer is None else layer - 1
    assert p.control_sensor_node == node
    assert state.control_C == state.temperature_C == state.node_temperatures_C[node]
    assert p.control_temperature_C(state) == state.control_C
    assert p.control_heat_capacity_Wh_per_K == pytest.approx(p.model.capacities_J_K[node] / 3600)
    assert [p.wp_supply_node, p.wp_return_node, p.heating_supply_node, p.heating_return_node] == [1, 9, 0, 8]


@pytest.mark.parametrize('value', [0, -1, 11, 1.5, True, False, '5'])
def test_invalid_config_rejected_before_initialization_or_run(value):
    c = configuration(value)
    for build in (build_simulation_params, initial_state):
        with pytest.raises(ValueError, match='control_sensor_layer'):
            build(c)


@pytest.mark.parametrize('method', ['heating_curve',
                                   'cop_optimized', 'pause_load_shifting', 'run_load_shifting', 'reinforcement_learning'])
@pytest.mark.parametrize('layer', [1, 5, 10])
def test_all_controllers_read_the_same_sensor_and_keep_physical_ports(method, layer):
    c = configuration(layer)
    c['control']['method'] = method
    instant = datetime(2025, 1, 1, 12)
    p = build_simulation_params(c)
    original = initial_state(c, 0)
    expected = original.storage.node_temperatures_C[layer - 1]
    import simulation.step as plant
    readings, return_readings, hot_readings = [], [], []
    def power(request, temperature, *args, **kwargs):
        readings.append(temperature)
        return original_power(request, temperature, *args, **kwargs)
    def cop(temperature, *args, **kwargs):
        readings.append(temperature)
        return original_cop(temperature, *args, **kwargs)
    def wp(return_C, *args, **kwargs):
        return_readings.append(return_C)
        return fixed_spread_heat_pump(return_C, *args, **kwargs)
    from models.radiator import RadiatorMap
    original_flow = RadiatorMap.flow
    def hk(radiator, hot_C, *args, **kwargs):
        hot_readings.append(hot_C)
        return original_flow(radiator, hot_C, *args, **kwargs)
    original_power, original_cop = plant.apply_power_request, plant.update_cop_control
    with patch.object(plant, 'apply_power_request', side_effect=power), \
         patch.object(plant, 'update_cop_control', side_effect=cop), \
         patch.object(plant, 'fixed_spread_heat_pump', side_effect=wp), \
         patch.object(RadiatorMap, 'flow', new=hk), \
         patch('control.rl_power.RLController') as controller:
        controller.return_value.predict.return_value = .8
        result = run_simulation(c, weather_points=[WeatherPoint(instant, 0)],
                                daily_temperatures={instant.date(): 0})
    assert readings and readings[0] == expected
    assert return_readings[0] == 20
    assert any(temperature == pytest.approx(60) for temperature in hot_readings)
    row = build_result_rows(result)[0]
    assert row['storage_control_sensor_layer'] == layer
    assert row['storage_temperature_C'] == row['storage_control_temperature_C'] == row[f'storage_node_{layer}_temperature_C']
    if method == 'heating_curve':
        assert row['heat_pump_requested_thermal_power_W'] == pytest.approx(
            heating_curve_power_request(original, SimulationInput(0, instant), p)[0].requested_heat_W)


def test_manual_state_rebinding_and_control_helpers_use_configuration_without_mutation():
    c = configuration(1)
    p = build_simulation_params(c)
    state = initial_state(c, 0)
    state.storage = StorageState(0, tuple(np.linspace(60, 20, 10)))  # Default sensor on this manually supplied state.
    original_temperature = state.storage.temperature_C
    result = simulate_step(state, SimulationInput(0), p)
    assert state.storage.temperature_C == original_temperature
    assert result.state.storage.control_sensor_layer == 1
    assert result.state.storage.temperature_C == result.state.storage.top_C
    assert result.heat_pump_requested_thermal_power_W == heating_curve_power_request(state, SimulationInput(0), p)[0].requested_heat_W


@pytest.mark.parametrize('layer', [1, 5, 10])
def test_rl_training_and_inference_observe_storage_profile(layer):
    pytest.importorskip('gymnasium')
    from learning.environment import HeatPumpEnv
    from datetime import timedelta
    c = configuration(layer)
    instant = datetime(2025, 1, 1, 12)
    weather = [WeatherPoint(instant + timedelta(minutes=2 * i), 0) for i in range(2)]
    env = HeatPumpEnv(c, weather, random_start=False)
    values, _ = env.reset()
    nodes = env.state.storage.node_temperatures_C  # independent of the selected control sensor
    assert values[4] == pytest.approx((nodes[0] - 45) / 20)
    assert values[6] == pytest.approx((nodes[-1] - 35) / 20)
    controller = RLController.__new__(RLController)
    controller.power_levels = (-1, 0, .4, .6, .8, 1)
    controller.params = env.params
    controller.model = Mock()
    controller.model.predict.return_value = (0, None)
    controller.predict(env.state, SimulationInput(0, instant))
    assert controller.model.predict.call_args.args[0] == pytest.approx(values)
