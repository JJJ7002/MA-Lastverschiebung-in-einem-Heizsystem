from pathlib import Path
import sys
from datetime import datetime, timedelta
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support import load_test_config
from simulation.engine import build_simulation_params, initial_state, run_simulation
from simulation.step import SimulationInput
from control.heating_curve import (heating_curve_power_request, heating_curve_target,
                                   heating_supply_curve_power_request)
from analysis.visualize import calculate_radiator_heating_curve
from weather.loader import WeatherPoint


def configuration():
    c = load_test_config()
    c['simulation'].update(start_time=None, end_time=None, settling_hours=0)
    c['control']['night_setback'].update(enabled=False)
    c['control']['heating_curve']['slope_factor'] = 1.0
    return c


@pytest.mark.parametrize('outside', [-15, 0, 10, 21, 25])
def test_base_curve_matches_required_stationary_radiator_supply(outside):
    c = configuration()
    c['control']['heating_curve']['offset_K'] = 0
    p = build_simulation_params(c)
    room = c['building']['T_set_C']
    q = max(0, p.building.stationary_UA_W_per_K * (room-outside))
    assert heating_curve_target(outside, None, p) == pytest.approx(
        min(p.heat_pump.sink_temp_max_C, room+p.radiator.temperatures(q)[0]))


def test_shift_and_slope_scale_excess_and_night_setpoint():
    c = configuration()
    c['control']['heating_curve'].update(offset_K=2, slope_factor=1.2)
    c['control']['night_setback'].update(enabled=True, warmup_minutes_pro_K=0)
    p = build_simulation_params(c)
    day = c['building']['T_set_C']  # night setback lowers the room target by 3 K
    for hour, room in [(12, day), (23, day - 3)]:
        excess = p.radiator.temperatures(p.building.stationary_UA_W_per_K*room)[0]
        reference = p.radiator.temperatures(max(0, p.building.stationary_UA_W_per_K*(room-20)))[0]
        assert heating_curve_target(0, datetime(2025,1,1,hour), p) == pytest.approx(
            room+reference+1.2*(excess-reference)+2)
    rows = calculate_radiator_heating_curve(c)
    row = next(r for r in rows if r['outside_temp_C'] == 0)
    assert row['room_target_C'] == day
    excess = p.radiator.temperatures(p.building.stationary_UA_W_per_K*day)[0]
    reference = p.radiator.temperatures(p.building.stationary_UA_W_per_K*(day-20))[0]
    assert row['buffer_target_C'] == pytest.approx(day+reference+1.2*(excess-reference)+2)


def test_slope_adjustment_rotates_curve_about_reference_point():
    c = configuration()
    c['control']['heating_curve']['offset_K'] = 0
    p0 = build_simulation_params(c)
    c['control']['heating_curve']['slope_factor'] = 1.0
    p1 = build_simulation_params(c)
    baseline_at_ref = heating_curve_target(20, None, p0)
    c['control']['heating_curve']['slope_factor'] = 1.5
    p2 = build_simulation_params(c)
    assert heating_curve_target(20, None, p2) == pytest.approx(baseline_at_ref)
    assert heating_curve_target(-10, None, p2) > heating_curve_target(-10, None, p0)
    assert heating_curve_target(25, None, p2) < heating_curve_target(25, None, p0)


@pytest.mark.parametrize('key,value', [('slope_factor',0),('slope_factor',-1),('slope_factor',float('nan')),('offset_K',float('inf'))])
def test_invalid_adjustments_rejected(key,value):
    c=configuration()
    c['control']['heating_curve'][key]=value
    with pytest.raises(ValueError):
        build_simulation_params(c)


def test_shared_power_formula_and_new_strategy_dispatch():
    c=configuration()
    p=build_simulation_params(c)
    s=initial_state(c,0)
    request,target=heating_curve_power_request(s,SimulationInput(0),p)
    loss=max(0, p.building.stationary_UA_W_per_K * (p.room_target_C_at(None) - 0))
    assert request.requested_heat_W == pytest.approx(max(0,loss+p.heating_curve_temperature_gain_W_per_K*(target-s.storage.control_C)))
    result=run_simulation(c,weather_points=[WeatherPoint(datetime(2025,1,1,12),0)])
    assert len(result.timestamps)==1
    for old in ['hysteresis','buffer_load_control']:
        c['control']['method']=old
        with pytest.raises(ValueError,match='control.method'):
            run_simulation(c,weather_points=[WeatherPoint(datetime(2025,1,1,12),0)])


def test_heating_curve_base_demand_is_stationary_and_independent_of_building_state():
    from dataclasses import replace

    c = configuration()
    p = build_simulation_params(c)
    inputs = SimulationInput(0, datetime(2025, 1, 1, 12))
    state = initial_state(c, 0)
    cold = replace(state, building=replace(state.building, T_room_C=17, T_mass_C=15))
    warm = replace(state, building=replace(state.building, T_room_C=23, T_mass_C=24))

    cold_request, cold_target = heating_curve_power_request(cold, inputs, p)
    warm_request, warm_target = heating_curve_power_request(warm, inputs, p)
    stationary = max(0, p.building.stationary_UA_W_per_K * (p.room_target_C_at(inputs.timestamp) - inputs.outside_temperature_C))
    correction = p.heating_curve_temperature_gain_W_per_K * (cold_target - cold.storage.control_C)

    assert cold_target == pytest.approx(warm_target)
    assert cold_request.building_loss_W == pytest.approx(stationary)
    assert warm_request.building_loss_W == pytest.approx(stationary)
    assert cold_request.requested_heat_W == pytest.approx(max(0, stationary + correction))
    assert warm_request.requested_heat_W == pytest.approx(max(0, stationary + correction))


def test_heating_curve_supply_uses_required_supply_plus_offset():
    c = configuration()
    c['control']['method'] = 'heating_curve_supply'
    c['control']['heating_curve_supply'] = {'offset_K': 4.0}
    c['storage']['T_storage_init_C'] = 30.0
    p = build_simulation_params(c)
    state = initial_state(c, 0)
    instant = datetime(2025, 1, 1, 12)
    inputs = SimulationInput(0, instant)
    request, target = heating_supply_curve_power_request(state, inputs, p)
    room_target = p.room_target_C_at(instant + timedelta(minutes=p.room_target_lookahead_minutes))
    demand = p.room_heat_demand_W(state.building.T_room_C, 0, room_target,
                                  state.building.T_mass_C)
    flow = p.heating_flow(state.storage.top_C, state.building.T_room_C,
                          demand, 0, instant)
    assert target == pytest.approx(flow.target_supply_C + 4.0)
    assert request.requested_heat_W >= 0

    result = run_simulation(c, weather_points=[WeatherPoint(instant, 0)])
    row = result.hydraulics[0]
    assert row['buffer_target_C'] == pytest.approx(target)
    assert row['heating_supply_temperature_C'] < row['heating_supply_target_C']


def test_heating_curve_supply_slope_factor_scales_required_supply_rise():
    c = configuration()
    c['control']['method'] = 'heating_curve_supply'
    c['storage']['T_storage_init_C'] = 30.0
    instant = datetime(2025, 1, 1, 12)
    targets = {}
    for slope in (1.0, 0.5):
        c['control']['heating_curve_supply'] = {'offset_K': 4.0, 'slope_factor': slope}
        p = build_simulation_params(c)
        state = initial_state(c, 0)
        _, targets[slope] = heating_supply_curve_power_request(state, SimulationInput(0, instant), p)
    p = build_simulation_params(c)
    room_target = p.room_target_C_at(instant + timedelta(minutes=p.room_target_lookahead_minutes))
    assert targets[0.5] - 4.0 - room_target == pytest.approx(0.5 * (targets[1.0] - 4.0 - room_target))
    for bad in (0, -1, float('nan')):
        c['control']['heating_curve_supply']['slope_factor'] = bad
        with pytest.raises(ValueError, match='slope_factor'):
            build_simulation_params(c)
