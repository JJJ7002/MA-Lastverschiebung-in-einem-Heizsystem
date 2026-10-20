from pathlib import Path
import sys
from dataclasses import replace
from datetime import datetime, timedelta
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.building import BuildingParams, BuildingState, advance_building
from support import load_test_config
from simulation.engine import build_simulation_params, initial_state, run_simulation, build_result_rows
from simulation.step import SimulationInput, simulate_step
from weather.loader import WeatherPoint


def parameters(**changes):
    return BuildingParams(UA_W_per_K=300, mean_room_height_m=2.5, **changes)


def configuration():
    c = load_test_config()
    c['simulation'].update(start_time=None, end_time=None, settling_hours=0)
    c['control']['night_setback']['enabled'] = False
    c['heat_pump'].update(min_on_time_min=0, min_off_time_min=0)
    return c


def test_reference_and_stationary_loss_split():
    p = parameters()
    assert p.C_air_Wh_per_K == 250
    assert p.C_mass_Wh_per_K == pytest.approx(21666.6666667)
    assert p.H_am_W_per_K == pytest.approx(2969.0542907)
    # UA is split 50/50 into air->outside and mass->outside.
    assert p.H_ae_W_per_K == pytest.approx(150)
    assert p.H_me_W_per_K == pytest.approx(150)
    s = p.initialize(21, 0)
    air, mass = p.losses(s, 0)
    # Each loss uses its own node temperature: the stationary mass is cooler than
    # the air, so its loss is smaller and the total is below UA * dT.
    assert air == pytest.approx(150 * 21)
    assert mass == pytest.approx(150 * s.T_mass_C)
    assert s.T_mass_C < 21 and mass < air
    assert air + mass == pytest.approx(p.stationary_UA_W_per_K * 21)
    assert p.stationary_UA_W_per_K == pytest.approx(
        150 + 1 / (1 / p.H_am_W_per_K + 1 / 150))
    assert p.heat_request(s, 0, 21, 30) == pytest.approx(air + mass)
    step = advance_building(s, 0, air + mass, p, 24)
    assert step.state.T_room_C == pytest.approx(21)
    assert step.state.T_mass_C == pytest.approx(s.T_mass_C)


@pytest.mark.parametrize('dt_h', [1/3600, 10/3600, 1, 24])
@pytest.mark.parametrize('outside,power', [(-15, 0), (0, 9000), (35, 0)])
def test_energy_and_signed_losses(dt_h, outside, power):
    p = parameters()
    s = BuildingState(20, 19)
    step = advance_building(s, outside, power, p, dt_h)
    assert step.balance_error_Wh == pytest.approx(0, abs=1e-8)
    assert step.energy_change_Wh == pytest.approx(
        (power-step.air_loss_W-step.mass_loss_W-step.temperature_cap_loss_W)*dt_h, abs=1e-8)
    if outside == 35:
        assert step.air_loss_W < 0 and step.mass_loss_W < 0


def test_configurable_upper_temperature_limit_caps_air_and_mass_with_energy_accounting():
    p = parameters(T_max_C=23.5)
    initialized = p.initialize(30, 35, 29)
    assert initialized.T_room_C == initialized.T_mass_C == 23.5

    step = advance_building(BuildingState(24, 24), 35, 10000, p, 24)
    assert step.state.T_room_C <= 23.5
    assert step.state.T_mass_C <= 23.5
    assert step.temperature_cap_loss_W > 0
    assert step.balance_error_Wh == pytest.approx(0, abs=1e-8)


def test_internal_exchange_conserves_energy_and_air_reacts_fast():
    p = replace(parameters(), UA_W_per_K=0)
    s = BuildingState(15, 25)
    step = advance_building(s, 0, 0, p, 1)
    assert step.energy_change_Wh == pytest.approx(0, abs=1e-8)
    assert 15 < step.state.T_room_C < step.state.T_mass_C < 25
    initial = BuildingState(20, 20)
    heated = advance_building(initial, 20, 6000, p, 10/3600).state
    assert heated.T_room_C-20 > 20*(heated.T_mass_C-20)


def test_room_regulation_uses_air_capacity_and_mass_exchange():
    p = parameters()
    s = BuildingState(20, 19)
    request = p.heat_request(s, 0, 21, 30)
    assert request == pytest.approx(150*20+p.H_am_W_per_K+250/.5)
    actual_air_derivative = (request-150*20-p.H_am_W_per_K)/250
    assert actual_air_derivative == pytest.approx(2)  # K/hour, tau = 0.5 h


@pytest.mark.parametrize('kwargs', [dict(mass_heat_capacity_Wh_per_m2K=0), dict(direct_loss_fraction=-.1),
    dict(direct_loss_fraction=1.1), dict(heated_floor_area_m2=-1), dict(mean_room_height_m=0),
    dict(T_max_C=float('inf'))])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        replace(parameters(), **kwargs)


def test_derived_air_capacity():
    with pytest.raises(ValueError, match='mean_room_height_m'):
        replace(parameters(), mean_room_height_m=None)
    assert replace(parameters(), heated_floor_area_m2=300, mean_room_height_m=2.5).C_air_Wh_per_K == 250
    assert replace(parameters(), mass_heat_capacity_Wh_per_m2K=90).C_mass_Wh_per_K == 27000
    p = parameters(direct_loss_fraction=1)
    assert p.H_me_W_per_K == 0


def test_initialization_requires_weather_and_respects_explicit_mass():
    c = configuration()
    with pytest.raises(ValueError, match='first outside'):
        initial_state(c)
    s = initial_state(c, -5)
    p = build_simulation_params(c).building
    room = c['building']['T_room_init_C']
    assert s.building.T_mass_C == pytest.approx((p.H_am_W_per_K*room+p.H_me_W_per_K*(-5))/(p.H_am_W_per_K+p.H_me_W_per_K))
    c['building']['T_mass_init_C'] = 18
    assert initial_state(c).building.T_mass_C == 18


def test_simulation_outputs_and_under_supply():
    c = configuration()
    c['storage']['T_storage_init_C'] = 15
    start = initial_state(c, 0)
    p = build_simulation_params(c)
    result = simulate_step(start, SimulationInput(0, datetime(2025,1,1)), p, -1)
    assert result.heating_power_to_building_W == 0
    assert result.state.building.T_room_C < start.building.T_room_C  # no heating: the room cools down
    assert abs(result.building_balance_error_Wh) < 1e-7
    weather = [WeatherPoint(datetime(2025,1,1)+timedelta(minutes=2*i), 0) for i in range(5)]
    sim = run_simulation(c, weather_points=weather)
    rows = build_result_rows(sim)
    assert rows[0]['mass_temperature_C'] is not None
    assert rows[0]['building_heat_loss_W'] == pytest.approx(rows[0]['building_air_loss_W']+rows[0]['building_mass_loss_W'])
    assert sim.building_parameters['C_air_Wh_per_K'] == pytest.approx(p.building.C_air_Wh_per_K)
    assert sim.mass_temperatures_C[-1] == rows[-1]['mass_temperature_C']


def test_two_node_time_step_converges():
    p = parameters()
    def run(dt):
        s = p.initialize(19, 0)
        for _ in range(round(1800/dt)):
            request = p.heat_request(s, 0, 21, 30)
            s = advance_building(s, 0, request, p, dt/3600).state
        return s.T_room_C
    coarse, fine, reference = run(20), run(5), run(1)
    assert abs(fine-reference) < abs(coarse-reference)
    assert abs(fine-reference) < .01


@pytest.mark.parametrize('method,settled_by', [
    ('heating_curve', 'heating_curve'),
    ('heating_curve_supply', 'heating_curve_supply'),
    ('cop_optimized', 'cop_optimized'),
    ('reinforcement_learning', 'reinforcement_learning'),
    ('pause_load_shifting', 'heating_curve_supply'),  # settled like its baseline run
    ('run_load_shifting', 'heating_curve_supply'),
])
def test_settling_uses_own_strategy_or_load_shifting_baseline(method, settled_by):
    from unittest.mock import patch
    weather = [WeatherPoint(datetime(2025,1,1)+timedelta(minutes=2*i), 0) for i in range(8)]
    daily = {datetime(2025,1,1).date(): 0}
    def run(selected, points, settling):
        c = configuration()
        c['control']['method'] = selected
        for name in ('pause_load_shifting', 'run_load_shifting'):
            c['control'].setdefault(name, {})['base_method'] = 'heating_curve_supply'  # not the yaml default
        c['simulation']['settling_hours'] = settling
        c['simulation']['start_time'] = points[3].timestamp if settling else None
        with patch('control.rl_power.RLController') as controller:
            controller.return_value.predict.return_value = .4
            return run_simulation(c, weather_points=points, daily_temperatures=daily)
    sim = run(method, weather, .1)  # 0.1 h = 3 settling steps of 2 min
    assert len(sim.timestamps) == 5
    assert sim.settling_hours == pytest.approx(.1)
    # Settling must end in the state reached by running the settling strategy itself.
    reference = run(settled_by, weather[:3], 0)
    assert sim.initial_building_temperatures_C == pytest.approx(
        {'air_C': reference.room_temperatures_C[-1], 'mass_C': reference.mass_temperatures_C[-1]})


def test_heating_curve_feedforward_is_stationary_but_recovery_and_rl_keep_mass_state():
    from control.heating_curve import heating_curve_power_request
    from control.heat_pump_power import heat_pump_power_request
    from control.pause_load_shifting import _recovered
    from control.rl_power import observation, OBSERVATION_VERSION, OBSERVATION_SIZE
    c = configuration()
    c['control']['method'] = 'heating_curve'
    p = build_simulation_params(c)
    state = initial_state(c, 0)
    cold = replace(state, building=replace(state.building, T_mass_C=17))
    inp = SimulationInput(0, datetime(2025,1,1,12))
    # heating_curve uses the stationary loss for its feedforward; the shared
    # helper without that override still reports the current two-node losses.
    assert heating_curve_power_request(cold, inp, p)[0].building_loss_W == pytest.approx(
        heating_curve_power_request(state, inp, p)[0].building_loss_W)
    assert heat_pump_power_request(48, cold, inp, p, 2, 2000).building_loss_W == pytest.approx(sum(p.building.losses(cold.building,0)))
    assert not _recovered(cold, state, p, .1, .1)
    a = observation(state, 0, inp.timestamp, p)
    b = observation(cold, 0, inp.timestamp, p)
    assert OBSERVATION_VERSION == 15 and len(a) == OBSERVATION_SIZE
    # The mass is no direct RL input (measured: no benefit); it only enters
    # indirectly through the required supply in the supply-margin feature (index 7).
    assert [x for i, x in enumerate(a) if i != 7] == [x for i, x in enumerate(b) if i != 7]


def test_whole_plant_energy_balance_includes_both_building_nodes():
    c = configuration()
    p = build_simulation_params(c)
    s = initial_state(c, -5)
    for index in range(12):
        storage_before = p.storage.model.energy_Wh(s.storage.node_temperatures_C)
        result = simulate_step(s, SimulationInput(-5, datetime(2025,1,1)+timedelta(minutes=2*index)), p, .5)
        storage_after = p.storage.model.energy_Wh(result.state.storage.node_temperatures_C)
        delta = storage_after-storage_before+result.building_energy_change_Wh
        net = (result.thermal_power_W-result.storage_loss_W-result.building_heat_loss_W)*p.dt_hours
        assert delta == pytest.approx(net, abs=1e-7)
        assert result.heating_power_to_building_W <= result.heating_requested_power_W+1e-6
        s = result.state


def test_warmup_settings_are_checked():
    c = configuration()
    weather = [WeatherPoint(datetime(2025,1,1), 0)]
    c['simulation']['start_time'] = weather[0].timestamp
    for hours in [-1, float('nan'), 168]:
        c['simulation']['settling_hours'] = hours
        with pytest.raises(ValueError, match='settling_hours'):
            run_simulation(c, weather_points=weather)


def test_settling_does_not_shorten_measurement_window(monkeypatch):
    import simulation.engine as engine
    c = configuration()
    start = datetime(2025, 1, 1)
    c['simulation'].update(start_time=start, end_time=start + timedelta(minutes=10))
    points = [WeatherPoint(start + timedelta(minutes=2*i), -5) for i in range(-3, 5)]
    baseline = run_simulation(c, weather_points=points)
    c['simulation']['settling_hours'] = .1
    requested_starts = []

    def load_weather(config, root):
        requested_starts.append(config['simulation']['start_time'])
        return points

    monkeypatch.setattr(engine, 'load_api_weather', load_weather)
    settled = run_simulation(c)
    assert requested_starts == [start - timedelta(minutes=6)]
    assert c['simulation']['start_time'] == start
    assert settled.timestamps == baseline.timestamps
    assert len(settled.timestamps) == 5
    assert len(settled.settling_history) == 3
    assert settled.settling_history[0]['timestamp'] == '2024-12-31 23:54:00'
    with pytest.raises(ValueError, match='Incomplete|incomplete'):
        run_simulation(c, weather_points=points[1:])


def test_settling_requires_start_and_rejects_old_parameter():
    c = configuration()
    points = [WeatherPoint(datetime(2025, 1, 1), 0)]
    c['simulation']['settling_hours'] = 1
    with pytest.raises(ValueError, match='explicit simulation.start_time'):
        run_simulation(c, weather_points=points)
    c['simulation']['warmup_hours'] = 1
    with pytest.raises(ValueError, match='Replace simulation.warmup_hours'):
        run_simulation(c, weather_points=points)


@pytest.mark.parametrize('hours,steps', [(0, 0), (.1, 3), (.11, 4)])
def test_warmup_carries_all_temperatures_and_excludes_energy(hours, steps, tmp_path):
    import csv
    from simulation.engine import save_settling_csv
    c = configuration()
    c['simulation']['settling_hours'] = hours
    c['storage']['T_storage_init_C'] = 30
    weather = [WeatherPoint(datetime(2025,1,1)+timedelta(minutes=2*i), -5) for i in range(10)]
    c['simulation']['start_time'] = weather[steps].timestamp
    c['simulation']['end_time'] = weather[-1].timestamp + timedelta(minutes=2)
    sim = run_simulation(c, weather_points=weather)
    assert len(sim.settling_history) == steps
    assert len(sim.timestamps) == len(weather)-steps
    assert sim.measurement_initial_state['timestamp'] == weather[steps].timestamp.isoformat(sep=' ')
    assert sim.timestamps[0] == sim.measurement_initial_state['timestamp']
    assert sim.timestamps[0] == c['simulation']['start_time'].isoformat(sep=' ')
    assert sim.timestamps[-1] == weather[-1].timestamp.isoformat(sep=' ')
    p = build_simulation_params(c)
    state = initial_state(c, -5)
    for point in weather[:steps]:
        state = simulate_step(state, SimulationInput(-5, point.timestamp), p).state
    assert sim.measurement_initial_state['air_temperature_C'] == pytest.approx(state.building.T_room_C)
    assert sim.measurement_initial_state['mass_temperature_C'] == pytest.approx(state.building.T_mass_C)
    assert sim.measurement_initial_state['storage_layer_temperatures_C'] == pytest.approx(state.storage.node_temperatures_C)
    assert sim.total_electric_energy_Wh == pytest.approx(sum(sim.electric_power_W)*p.dt_hours)
    assert sim.total_thermal_energy_Wh == pytest.approx(sum(sim.thermal_power_W)*p.dt_hours)
    path = tmp_path / 'settling_results.csv'
    save_settling_csv(sim, path)
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == steps
    if steps:
        assert float(rows[-1]['storage_layer_1_C']) == pytest.approx(state.storage.top_C)
        assert float(rows[-1]['mass_temperature_C']) == pytest.approx(state.building.T_mass_C)
        assert sim.settling_history[-1]['air_temperature_C'] == pytest.approx(state.building.T_room_C)
