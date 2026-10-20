from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.performance_map import PerformanceMap
from support import load_test_config
from simulation.engine import build_simulation_params, initial_state
from simulation.step import simulate_step, SimulationInput
from models.heat_pump import get_performance
from datetime import datetime, timedelta
from unittest.mock import patch
from weather.loader import WeatherPoint
from simulation.engine import run_simulation

ROOT = Path(__file__).resolve().parents[1]


def test_export_complete_and_monotonic():
    model = PerformanceMap(ROOT / 'data/wpl25a_toolbox.csv')
    assert len(model.grid) == 13608
    assert model.sinks == list(range(30, 66, 5))
    for source in range(-20, 41):
        for sink in model.sinks:
            curve = model.curves(source, sink)
            assert all(b[0] >= a[0] for a, b in zip(curve, curve[1:]))
            for command, (q, cop) in zip(model.power_fractions, curve):
                actual_cop, actual_q, electric = model.performance(source, sink, command)
                assert actual_q == pytest.approx(q)
                assert actual_cop == pytest.approx(cop, abs=.015)  # duplicate rounded powers
                assert electric * actual_cop == pytest.approx(q)


def test_temperature_interpolation_and_subminimum():
    model = PerformanceMap(ROOT / 'data/wpl25a_toolbox.csv')
    a = model.curves(0, 35)[-1]
    b = model.curves(1, 35)[-1]
    c = model.curves(0, 40)[-1]
    d = model.curves(1, 40)[-1]
    assert model.curves(.5, 37.5)[-1] == pytest.approx(tuple(sum(v)/4 for v in zip(a,b,c,d)))
    cop, q, electric = model.performance(7, 35, 0)
    assert cop == model.curves(7,35)[0][1]
    assert q == pytest.approx(model.curves(7,35)[0][0])
    assert electric == pytest.approx(q/cop)
    assert model.performance(7,35,-1) == (0,0,0)
    assert model.performance(7,35,.1)[1] == pytest.approx(model.curves(7,35)[2][0])
    assert model.curves(-30,20) == model.curves(-20,30)


def test_default_simulation_uses_modulated_map():
    config = load_test_config()
    config['heat_pump']['min_off_time_min'] = 0
    config['control']['night_setback']['enabled'] = False
    config['storage']['T_storage_init_C'] = 30
    config['storage']['V_storage_L'] = 1e12  # Isolate map interpolation at a constant return.
    params = build_simulation_params(config)
    state = initial_state(config, 7)
    output = simulate_step(state, SimulationInput(7), params, requested_power_fraction=.8)
    expected = get_performance(7, 30 + params.heat_pump.supply_spread_K, params.heat_pump, .8)
    assert (output.cop, output.thermal_power_W, output.electric_power_W) == pytest.approx(expected)


def test_subminimum_request_delivers_minimum_until_controller_shuts_down():
    config = load_test_config()
    config['heat_pump']['min_volume_flow_L_h'] = 0  # Isolate compressor minimum from hydraulic limits.
    config['heat_pump'].update(min_on_time_min=0, min_off_time_min=0)
    config['control']['night_setback']['enabled'] = False
    config['storage']['T_storage_init_C'] = 30
    config['storage']['V_storage_L'] = 1e12  # Constant return isolates manufacturer minimum output.
    params = build_simulation_params(config)
    state = initial_state(config, 7)
    first = simulate_step(state, SimulationInput(7), params, requested_power_fraction=0)
    supply = 30 + params.heat_pump.supply_spread_K  # Constant return + configured spread.
    minimum, cop = params.heat_pump.performance_map.curves(7, supply)[0]
    maximum = params.heat_pump.performance_map.curves(7, supply)[-1][0]
    assert first.state.heat_pump.is_on
    assert first.thermal_power_W == pytest.approx(minimum)
    assert first.cop == pytest.approx(cop)
    assert first.power_fraction == 0
    assert first.heat_pump_relative_thermal_power == pytest.approx(minimum/maximum)
    held = simulate_step(first.state, SimulationInput(10), params)
    assert held.state.commanded_power_fraction == first.state.commanded_power_fraction
    assert held.thermal_power_W == pytest.approx(
        params.heat_pump.performance_map.curves(10, supply)[0][0])
    from dataclasses import replace
    due = replace(held.state, steps_elapsed=params.control_steps)
    off = simulate_step(due, SimulationInput(10), params, requested_power_fraction=-1)
    assert not off.state.heat_pump.is_on
    assert off.thermal_power_W == off.electric_power_W == 0
    assert off.power_fraction == -1


@pytest.mark.parametrize('method', ['heating_curve', 'cop_optimized',
                                   'pause_load_shifting', 'run_load_shifting', 'reinforcement_learning'])
def test_all_controllers_use_same_map(method):
    config = load_test_config()
    config['heat_pump']['min_volume_flow_L_h'] = 0
    config['storage']['V_storage_L'] = 1e12  # Constant return isolates each controller's map use.
    config['weather']['source'] = 'csv'
    config['simulation'].update(start_time=None, end_time=None)
    config['control']['method'] = method
    config['control']['night_setback']['enabled'] = False
    config['heat_pump'].update(min_on_time_min=0, min_off_time_min=0)
    weather = [WeatherPoint(datetime(2020,1,1,12)+timedelta(minutes=2*i), -7+i/5)
               for i in range(30)]
    with patch('simulation.engine.load_daily_temperatures', return_value={datetime(2020,1,1).date(): -4}), \
         patch('control.rl_power.RLController') as controller:
        controller.return_value.predict.return_value = .8
        result = run_simulation(config, weather_points=weather)
    params = build_simulation_params(config)
    storage = config['storage']['T_storage_init_C']
    running = False
    for i, point in enumerate(weather):
        fraction = result.heat_pump_power_fractions[i]
        if fraction >= 0:
            running = True
            expected = get_performance(point.outside_temperature_C,
                                       config['storage']['T_storage_init_C'] + params.heat_pump.supply_spread_K,
                                       params.heat_pump, fraction)
            assert (result.cop[i],result.thermal_power_W[i],result.electric_power_W[i]) == pytest.approx(expected)
        else:
            assert result.thermal_power_W[i] == result.electric_power_W[i] == 0
        storage = result.storage_temperatures_C[i]
    assert running


def test_heat_pump_owns_switching_times():
    from dataclasses import replace
    from simulation.engine import build_simulation_params
    config = load_test_config()
    config['heat_pump'].update(min_on_time_min=17.5, min_off_time_min=12.5)
    params = build_simulation_params(config)
    assert params.heat_pump.min_on_time_min == 17.5
    assert params.heat_pump.min_off_time_min == 12.5
    assert not hasattr(params.operating, 'min_on_time_min')
    for key in ('min_on_time_min', 'min_off_time_min'):
        for value in (-1, float('nan'), float('inf')):
            with pytest.raises(ValueError, match=key):
                replace(params.heat_pump, **{key: value})
