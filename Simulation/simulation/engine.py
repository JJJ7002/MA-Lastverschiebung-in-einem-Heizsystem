from __future__ import annotations
from models.radiator import RadiatorMap
import yaml

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from datetime import datetime, timedelta
from math import ceil, isclose, isfinite, pi

from control.operating_constraints import OperatingParams, OperatingState, ScheduleOverride
from control.temperature_targets import HeatingCurveParams, RadiatorHeatingCurve, NightSetback
from weather.loader import load_weather_csv, resample_weather
from weather.meteoswiss import weather_source, load_api_weather, load_api_daily
from control.cop_optimized import CopOptimizedParams
from weather.loader import load_daily_temperatures
from models.heat_pump import get_performance
from models.building import BuildingParams
from models.heat_pump import HeatPumpParams
from models.storage import StorageParams, StorageState, resolve_sensor_node
from simulation.step import SimulationInput, SimulationParams, SimulationState, simulate_step


@dataclass(frozen=True)
class SimulationResult:
    timestamps: list[str]
    outside_temperatures_C: list[float]
    room_temperatures_C: list[float]
    storage_temperatures_C: list[float]
    heat_pump_on: list[bool]
    cop: list[float]
    thermal_power_W: list[float]
    electric_power_W: list[float]
    total_electric_energy_Wh: float
    total_thermal_energy_Wh: float
    night_setback: NightSetback = NightSetback()
    target_room_temperature_C: float = 21.0
    comfort_buffer_K: float = 1.0
    heat_pump_power_fractions: list[float] = field(default_factory=list)
    heat_pump_relative_thermal_powers: list[float] = field(default_factory=list)
    heating_power_to_building_W: list[float] = field(default_factory=list)
    building_heat_loss_W: list[float] = field(default_factory=list)
    reference_cop: list[float | None] = field(default_factory=list)
    cop_advantage: list[float | None] = field(default_factory=list)
    potential_cop: list[float] = field(default_factory=list)
    fallback_active: list[bool] = field(default_factory=list)
    fallback_override: list[bool] = field(default_factory=list)
    load_shifting_summary: dict[str, Any] = field(default_factory=dict)
    storage_node_temperatures_C: list[tuple[float, ...]] = field(default_factory=list)
    hydraulics: list[dict[str, Any]] = field(default_factory=list)
    heat_pump_starts: list[int] = field(default_factory=list)
    heat_pump_runtime_fractions: list[float] = field(default_factory=list)
    storage_mean_temperatures_C: list[float] = field(default_factory=list)
    mass_temperatures_C: list[float | None] = field(default_factory=list)
    building_parameters: dict[str, Any] = field(default_factory=dict)
    initial_building_temperatures_C: dict[str, Any] = field(default_factory=dict)
    settling_history: list[dict[str, Any]] = field(default_factory=list)
    measurement_initial_state: dict[str, Any] = field(default_factory=dict)
    settling_hours: float = 0.0
    initial_storage_energy_Wh: float = 0.0
    initial_building_energy_Wh: float = 0.0
    segment_starts: list[int] = field(default_factory=list)  # first step index of each heating-season segment
    storage_energy_change_Wh: float | None = None  # set when seasonal shutdown splits the run into segments

    @property
    def seasonal_cop(self) -> float:
        if self.total_electric_energy_Wh == 0:
            return 0.0
        return self.total_thermal_energy_Wh / self.total_electric_energy_Wh


def _build_heat_pump_params(config: dict[str, Any]) -> HeatPumpParams:
    from models.performance_map import load_performance_map
    map_path = config['heat_pump'].get('performance_map_file')
    if not map_path:
        raise ValueError('heat_pump.performance_map_file is required')
    return HeatPumpParams(
        source_temp_min_C=float(config['heat_pump']['T_source_min_C']),
        sink_temp_max_C=float(config['heat_pump']['T_sink_max_C']),
        min_on_time_min=float(config['heat_pump']['min_on_time_min']),
        min_off_time_min=float(config['heat_pump']['min_off_time_min']),
        max_volume_flow_L_h=(float(config['heat_pump']['max_volume_flow_L_h'])
                            if config['heat_pump'].get('max_volume_flow_L_h') is not None else None),
        minimum_supply_spread_K=float(config['heat_pump'].get(
            'minimum_supply_spread_K', config['heat_pump'].get('minimum_temperature_lift_K', 0.01))),
        min_volume_flow_L_h=float(config['heat_pump'].get('min_volume_flow_L_h', 1000)),
        supply_spread_K=float(config['heat_pump'].get(
            'supply_spread_K', config['heat_pump'].get('temperature_lift_K', 5))),
        performance_map=load_performance_map(str((Path(__file__).resolve().parents[1] / map_path).resolve())),
    )


def _parse_datetime(value: Any) -> datetime | None:
    if value in {None, ""}:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported datetime format: {value}")


def build_simulation_params(config: dict[str, Any]) -> SimulationParams:
    curve_settings = config['control'].get('heating_curve', {})
    radiator = (RadiatorMap(volume_flow_L_h=float(config['heating'].get('volume_flow_L_h', 1500)),
                           **config['heating']['radiator']) if 'radiator' in config['heating'] else None)
    heating_curve = config['control'].get('heating_circuit_curve') or dict(slope=curve_settings.get('slope', -1.4), offset_C=curve_settings.get('offset_C', 48.5))
    buffer_curve = (RadiatorHeatingCurve(radiator, BuildingParams.from_config(config['building']).stationary_UA_W_per_K,
                                        float(config['building']['T_set_C']),
                                        float(curve_settings.get('slope_factor', 1)),
                                        float(curve_settings.get('offset_K', 0)),
                                        float(curve_settings.get('reference_outside_C', 20)))
                    if radiator is not None else HeatingCurveParams(float(heating_curve['slope']), float(heating_curve['offset_C'])))
    storage_config = config['storage']
    storage_profile = _load_storage_profile(storage_config)
    params = SimulationParams(
        heating_curve_control=config['control'].get('method', 'heating_curve') in (
            'heating_curve', 'heating_curve_supply', 'pause_load_shifting', 'run_load_shifting',
            'reinforcement_learning_offset'),
        heating_supply_curve_control=(
            config['control'].get('method') == 'heating_curve_supply'
            or (config['control'].get('method') in LOAD_SHIFTING_METHODS
                and _load_shifting_base(config) == 'heating_curve_supply')),
        cop_fallback_supply_curve=(config['control'].get('cop_optimized', {}).get('fallback', 'heating_curve_supply')
                                   == 'heating_curve_supply'),
        heating_supply_curve_offset_K=float(config['control'].get('heating_curve_supply', {}).get('offset_K', 0.0)),
        heating_supply_curve_slope_factor=float(config['control'].get('heating_curve_supply', {}).get('slope_factor', 1.0)),
        heating_curve_temperature_gain_W_per_K=float(curve_settings.get('temperature_gain_W_per_K', 2000)),
        heating_supply_curve_temperature_gain_W_per_K=float(
            config['control'].get('heating_curve_supply', {}).get('temperature_gain_W_per_K', 2000)),
        heating_curve_temperature_hysteresis_K=float(curve_settings.get('temperature_hysteresis_K', 2)),
        heating_supply_curve_temperature_hysteresis_K=float(
            config['control'].get('heating_curve_supply', {}).get('temperature_hysteresis_K', 2)),
        hk_shift_outside_C=tuple(float(v) for v in (curve_settings.get('hk_shift_gain') or {}).get('outside_C', (0.0,))),
        hk_shift_gain_K_per_K=tuple(float(v) for v in (curve_settings.get('hk_shift_gain') or {}).get('K_per_K', (1.0,))),
        heating_circuit_curve=HeatingCurveParams(float(heating_curve['slope']), float(heating_curve['offset_C'])),
        heating_curve_reference=(config['control'].get('heating_circuit_curve') or {}).get('reference', 'supply'),
        dt_hours=(float(config["simulation"]["dt_minutes"]) / 60),
        step_minutes=float(config["simulation"]["dt_minutes"]),
        control_interval_minutes=(float(config['control']['interval_minutes'])
                                  if 'interval_minutes' in config['control'] else None),
        heating_volume_flow_L_h=float(config['heating'].get('volume_flow_L_h', 1500)),
        room_response_minutes=float(config['heating'].get('room_response_minutes', 30)),
        room_target_lookahead_minutes=float(config['heating'].get('room_target_lookahead_minutes', 0)),
        building=BuildingParams.from_config(config["building"]),
        storage=StorageParams(
            volume_L=float(storage_config.get('V_storage_L') or storage_profile['volume_L']),
            ua_W_per_K=float(storage_config.get('UA_storage_W_per_K') or storage_profile['ua_W_per_K']),
            node_count=storage_config.get('node_count', 10),
            control_sensor_layer=storage_config.get('control_sensor_layer'),
            diameter_m=float(storage_config.get('diameter_m') or storage_profile['diameter_m']),
        ),
        heat_pump=_build_heat_pump_params(config),
        k_radiator_W_per_K=float(config["heating"].get("k_radiator_W_per_K", 350)),
        radiator=radiator,
        target_room_temperature_C=float(config["building"]["T_set_C"]),
        operating=OperatingParams(
            heating_curve=buffer_curve,
            schedule=ScheduleOverride.from_config(config['control'].get('schedule', {})),
            night_setback=NightSetback.from_config(config['control'].get('night_setback', {})),
        ),
    )

    return params


def _load_storage_profile(storage_config: dict[str, Any]) -> dict[str, float]:
    """Resolve a named manufacturer tank profile; explicit legacy values override it."""
    model = storage_config.get('model', 'SBP 400 E')
    if not isinstance(model, str) or not model.strip():
        raise ValueError('storage.model must be a non-empty SBP E model name')
    profile_path = Path(__file__).resolve().parents[1] / 'data' / 'sbp_e_models.yaml'
    with profile_path.open('r', encoding='utf-8') as fh:
        profiles = yaml.safe_load(fh)
    try:
        profile = profiles[model.strip()]
    except (KeyError, TypeError):
        choices = ', '.join(profiles)
        raise ValueError(f"Unknown storage.model {model!r}; choose one of: {choices}") from None
    volume_L = float(profile['volume_L'])
    diameter_value = profile.get('tank_diameter_m')
    standby_kWh = float(profile['standby_loss_kWh_per_24h_at_65C'])
    if diameter_value is None:
        raise ValueError(f"storage.model {model!r} has no vessel diameter in {profile_path}")
    diameter_m = float(diameter_value)
    height_m = volume_L / 1000 / (pi * diameter_m**2 / 4)
    cylinder_area_m2 = pi * diameter_m * height_m + pi * diameter_m**2 / 2
    ua_W_per_K = standby_kWh * 1000 / (24 * (65 - 20))
    return {**profile, 'ua_W_per_K': ua_W_per_K, 'diameter_m': diameter_m,
            'equivalent_cylinder_area_m2': cylinder_area_m2}


def initial_state(config: dict[str, Any], outside_temperature_C=None) -> SimulationState:
    count = config['storage'].get('node_count', 10)
    sensor_layer = resolve_sensor_node(count, config['storage'].get('control_sensor_layer')) + 1
    nodes = config['storage'].get('initial_node_temperatures_C')
    if nodes is None:
        nodes = [float(config['storage']['T_storage_init_C'])] * count
    if len(nodes) != count:
        raise ValueError('Initial node temperatures must match storage.node_count')
    return SimulationState(
        building=BuildingParams.from_config(config["building"]).initialize(
            float(config["building"]["T_room_init_C"]), outside_temperature_C,
            config["building"].get("T_mass_init_C")),
        storage=StorageState(temperature_C=float(config["storage"]["T_storage_init_C"]),
                             node_temperatures_C=tuple(float(v) for v in nodes),
                             control_sensor_layer=sensor_layer),
        heat_pump=OperatingState(is_on=False, minutes_in_current_state=0),
    )


def heating_segments(points, schedule) -> list[list]:
    """Split weather points into runs in which heating is allowed.

    Without ``seasonal_heating_off`` (or when it never applies) this is one segment with all points,
    so the simulation is unchanged. Points inside the seasonal shutdown are skipped, not simulated.
    """
    points = list(points)
    if not schedule.seasonal_heating_off_enabled:
        return [points] if points else []
    segments, current = [], []
    for point in points:
        if schedule.heating_off_at(point.timestamp):
            if current:
                segments.append(current)
                current = []
        else:
            current.append(point)
    if current:
        segments.append(current)
    return segments


LOAD_SHIFTING_METHODS = ('pause_load_shifting', 'run_load_shifting')
LOAD_SHIFTING_BASE_METHODS = ('heating_curve', 'heating_curve_supply')


def _load_shifting_base(config: dict[str, Any]) -> str:
    """Base control of a load-shifting method: its `base_method` option, default heating_curve_supply."""
    method = config['control']['method']
    base = (config['control'].get(method) or {}).get('base_method', 'heating_curve_supply')
    if base not in LOAD_SHIFTING_BASE_METHODS:
        raise ValueError(f'control.{method}.base_method must be one of {LOAD_SHIFTING_BASE_METHODS}')
    return base


def _with_trial_scorer(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Load-shifting options plus the scorer that rates trials like the run's total score."""
    from analysis.scoring import ScoringParams
    from control.trial_scoring import TrialScorer
    options = dict(config['control'].get(name, {}))
    options.setdefault('scorer', TrialScorer(ScoringParams.from_config(config['scoring']),
                                             float(config['building']['comfort_buffer_K'])))
    return options


def _merge_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine load-shifting summaries of several segments: sum numbers, join lists."""
    merged: dict[str, Any] = {}
    for summary in summaries:
        for key, value in summary.items():
            if key not in merged:
                merged[key] = list(value) if isinstance(value, list) else value
            elif isinstance(value, list):
                merged[key] = merged[key] + value
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                merged[key] = merged[key] + value
    return merged


def run_simulation(config: dict[str, Any], *, weather_points=None, daily_temperatures=None, progress_callback=None) -> SimulationResult:
    project_root = Path(__file__).resolve().parents[1]
    output_interval_steps(config)
    if progress_callback:
        progress_callback('Wetterdaten laden', 0, 0)
    params = build_simulation_params(config)
    if 'warmup_hours' in config['simulation']:
        raise ValueError('Replace simulation.warmup_hours with simulation.settling_hours')
    settling_hours = float(config['simulation'].get('settling_hours', 0))
    if not isfinite(settling_hours) or settling_hours < 0:
        raise ValueError('simulation.settling_hours must be finite and nonnegative')
    settling_count = ceil(settling_hours / params.dt_hours)
    start_time = _parse_datetime(config['simulation'].get('start_time'))
    end_time = _parse_datetime(config['simulation'].get('end_time'))
    if settling_count and start_time is None:
        raise ValueError('simulation.settling_hours requires an explicit simulation.start_time')
    load_start = start_time - timedelta(hours=settling_count * params.dt_hours) if start_time else None
    weather_config = dict(config, simulation=dict(config['simulation'], start_time=load_start))
    if weather_points is not None:
        weather = list(weather_points)
    elif weather_source(config) == 'meteoswiss':
        weather = load_api_weather(weather_config, project_root)
    else:
        weather = load_weather_csv(project_root / config['simulation']['weather_file'])
    weather = resample_weather(weather, params.dt_hours)
    if start_time is not None or end_time is not None:
        weather = [
            point
            for point in weather
            if (load_start is None or point.timestamp >= load_start)
            and (end_time is None or point.timestamp < end_time)
        ]
    if not weather:
        raise ValueError("No weather points left after applying the selected time window.")
    settling_weather = [point for point in weather if start_time and point.timestamp < start_time]
    weather = [point for point in weather if start_time is None or point.timestamp >= start_time]
    if not weather:
        raise ValueError('No weather points in the measurement window')
    expected_times = [load_start + timedelta(hours=i * params.dt_hours) for i in range(settling_count)]
    settling_timestamps = [point.timestamp for point in settling_weather]
    if (len(settling_timestamps) != len(expected_times)
            or any(abs((actual - expected).total_seconds()) > 1e-6
                   for actual, expected in zip(settling_timestamps, expected_times))):
        raise ValueError('Insufficient or incomplete weather data before start_time for simulation.settling_hours')

    method = config['control'].get('method', 'heating_curve')
    if method not in ('heating_curve', 'heating_curve_supply', 'cop_optimized', 'reinforcement_learning', 'reinforcement_learning_offset', 'pause_load_shifting', 'run_load_shifting'):
        raise ValueError('control.method must be heating_curve, heating_curve_supply, cop_optimized, reinforcement_learning, reinforcement_learning_offset, pause_load_shifting or run_load_shifting')
    if method == 'pause_load_shifting':
        from control.pause_load_shifting import validate_pause_load_shifting
        validate_pause_load_shifting(config['control'].get('pause_load_shifting', {}))
    if method == 'run_load_shifting':
        from control.run_load_shifting import validate_run_load_shifting
        validate_run_load_shifting(config['control'].get('run_load_shifting', {}))
    controller = None
    if method == 'reinforcement_learning':
        from control.rl_power import RLController
        controller = RLController(config)
    if method == 'reinforcement_learning_offset':
        from control.rl_offset import ConstantShiftController, RLOffsetController
        constant = (config['control'].get('reinforcement_learning_offset') or {}).get('constant_shift_K')
        controller = (RLOffsetController(config) if constant is None
                      else ConstantShiftController(float(constant)))
    if method == 'cop_optimized':
        cop_config = config['control'].get('cop_optimized', {})
        if cop_config.get('fallback', 'heating_curve_supply') not in ('heating_curve_supply', 'heating_curve'):
            raise ValueError('control.cop_optimized.fallback must be heating_curve_supply or heating_curve')
        if daily_temperatures is not None:
            daily = daily_temperatures
        elif weather_source(config) == 'meteoswiss':
            daily = load_api_daily(config, project_root)
        else:
            daily = load_daily_temperatures(project_root / cop_config.get(
                'daily_weather_file', 'data/weather/ogd-smn_bas_d_historical.csv'))
        references = {}
        # Settling runs the COP control too, so its days need a reference as well.
        for day in sorted({point.timestamp.date() for point in settling_weather + weather}):
            if day not in daily:
                raise ValueError(f'Missing tre200d0 daily mean for {day}')
            target = min(params.heat_pump.sink_temp_max_C,
                         params.operating.heating_curve.target_temperature_C(daily[day])
                         + params.heat_pump.supply_spread_K)
            references[day] = get_performance(daily[day], target, params.heat_pump)[0]
        params.cop_control = CopOptimizedParams(
            references,
            full_power_advantage=float(cop_config.get('full_power_advantage', .2)),
            min_power_advantage=float(cop_config.get('min_power_advantage', 0.0)),
        )

    all_points = settling_weather + weather  # contiguous: settling prefix + measurement window
    segments = heating_segments(weather, params.operating.schedule)
    if not segments:
        raise ValueError('control.schedule.seasonal_heating_off covers the whole measurement window')
    position = {id(point): number for number, point in enumerate(all_points)}
    total_progress = settling_count * len(segments) + sum(len(segment) for segment in segments)
    progress_done = 0
    if progress_callback:
        progress_callback('Simulation', 0, total_progress)
    # Every strategy starts each segment from the same physical plant and is settled with its own
    # control: heating_curve, heating_curve_supply, cop_optimized and reinforcement_learning with
    # themselves, pause_ and run_load_shifting with their base method (base_method, default
    # heating_curve_supply). Settling energy is excluded from scores. The first segment keeps the
    # original settling; after a skipped shutdown the plant restarts and is settled with heating enabled.
    settling_params = params
    restart_params = replace(settling_params, operating=replace(
        params.operating, schedule=replace(params.operating.schedule, seasonal_heating_off_enabled=False)))
    settling_history = []
    timestamps: list[str] = []
    outside_temperatures_C: list[float] = []
    room_temperatures_C: list[float] = []
    storage_temperatures_C: list[float] = []
    heat_pump_on: list[bool] = []
    cop_values: list[float] = []
    potential_cops: list[float] = []
    thermal_power_values: list[float] = []
    electric_power_values: list[float] = []
    power_fractions: list[float] = []
    building_heating: list[float] = []
    building_losses: list[float] = []
    reference_cops, cop_advantages = [], []
    fallback_active, fallback_override = [], []
    storage_nodes, hydraulic_rows, starts, runtime_fractions, storage_means = [], [], [], [], []

    step_results = []
    simulated_points = []
    segment_starts = []
    summaries = []
    segment_electric_Wh = segment_thermal_Wh = 0.0
    storage_changes = []
    for number, segment in enumerate(segments):
        first = position[id(segment[0])]
        if number == 0 and segment[0] is weather[0]:
            settle_points, settle_params = settling_weather, settling_params
        else:
            settle_points = all_points[max(0, first - settling_count):first] if settling_count else []
            settle_params = restart_params
        state = initial_state(config, (settle_points or segment)[0].outside_temperature_C)
        if settle_points:
            for point in settle_points:
                settle_inputs = SimulationInput(point.outside_temperature_C, point.timestamp)
                # The offset controller is settled with the plain heating curve, exactly like in training.
                settle_request = (controller.predict(state, settle_inputs)
                                  if controller is not None and method != 'reinforcement_learning_offset'
                                  and state.steps_elapsed % params.control_steps == 0
                                  else None)
                settling_step = simulate_step(state, settle_inputs, settle_params, settle_request)
                state = settling_step.state
                row = dict(timestamp=point.timestamp.isoformat(sep=' '),
                           elapsed_hours=(len(settling_history) + 1) * params.dt_hours,
                           outside_temperature_C=point.outside_temperature_C,
                           air_temperature_C=state.building.T_room_C,
                           mass_temperature_C=state.building.T_mass_C,
                           storage_mean_temperature_C=state.storage.mean_temperature_C,
                           storage_top_temperature_C=state.storage.top_C,
                           storage_bottom_temperature_C=state.storage.bottom_C)
                row.update({f'storage_layer_{i + 1}_C': temperature
                            for i, temperature in enumerate(state.storage.node_temperatures_C)})
                settling_history.append(row)
                progress_done += 1
                if progress_callback:
                    progress_callback('Einschwingen', progress_done, total_progress)
            state = replace(state, total_electric_energy_Wh=0., total_thermal_energy_Wh=0., steps_elapsed=0,
                            buffer_target_C=None, requested_thermal_power_W=None,
                            building_loss_feedforward_W=None, buffer_temperature_correction_W=None,
                            shutdown_check_heat_W=None, minimum_heat_W=None)
        if number == 0:
            measurement_initial_state = dict(
                timestamp=segment[0].timestamp.isoformat(sep=' '),
                air_temperature_C=state.building.T_room_C,
                mass_temperature_C=state.building.T_mass_C,
                storage_layer_temperatures_C=list(state.storage.node_temperatures_C),
                storage_mean_temperature_C=state.storage.mean_temperature_C)
            initial_building = state.building
            initial_storage_energy = params.storage.model.energy_Wh(state.storage.node_temperatures_C)
        segment_initial_storage_energy = params.storage.model.energy_Wh(state.storage.node_temperatures_C)

        segment_starts.append(len(step_results))
        segment_results = []
        if method == 'pause_load_shifting':
            from control.pause_load_shifting import run_pause_load_shifting
            segment_results, summary = run_pause_load_shifting(
                segment, state, params, _with_trial_scorer(config, 'pause_load_shifting'))
            summaries.append(summary)
        elif method == 'run_load_shifting':
            from control.run_load_shifting import simulate_run_load_shifting
            segment_results, summary = simulate_run_load_shifting(
                segment, state, params, _with_trial_scorer(config, 'run_load_shifting'))
            summaries.append(summary)
        else:
            for point in segment:
                inputs = SimulationInput(outside_temperature_C=point.outside_temperature_C,
                                         timestamp=point.timestamp)
                request = None
                if controller is not None and state.steps_elapsed % params.control_steps == 0:
                    if method == 'reinforcement_learning_offset':
                        state = replace(state, hk_shift_K=controller.shift_K(state, inputs))
                    else:
                        request = controller.predict(state, inputs)
                result = simulate_step(
                    state=state,
                    inputs=inputs,
                    params=params,
                    requested_power_fraction=request,
                )
                state = result.state
                segment_results.append(result)
                progress_done += 1
                if progress_callback:
                    progress_callback('Simulation', progress_done, total_progress)
        if method in LOAD_SHIFTING_METHODS:
            progress_done += len(segment)
            if progress_callback:
                progress_callback('Simulation', progress_done, total_progress)
        step_results.extend(segment_results)
        simulated_points.extend(segment)
        final_state = segment_results[-1].state
        segment_electric_Wh += final_state.total_electric_energy_Wh
        segment_thermal_Wh += final_state.total_thermal_energy_Wh
        storage_changes.append(params.storage.model.energy_Wh(final_state.storage.node_temperatures_C)
                               - segment_initial_storage_energy)


    for point, result in zip(simulated_points, step_results):
        state = result.state

        timestamps.append(point.timestamp.isoformat(sep=" "))
        outside_temperatures_C.append(point.outside_temperature_C)
        room_temperatures_C.append(state.building.T_room_C)
        storage_temperatures_C.append(state.storage.temperature_C)
        storage_means.append(state.storage.mean_temperature_C)
        storage_nodes.append(state.storage.node_temperatures_C)
        starts.append(result.heat_pump_start_count)
        runtime_fractions.append(result.heat_pump_runtime_fraction)
        hydraulic_rows.append({key: getattr(result, key) for key in (
            'buffer_target_C', 'wp_supply_target_temperature_C', 'wp_supply_temperature_C', 'wp_return_temperature_C', 'wp_active_volume_flow_L_h',
            'wp_volume_flow_L_h', 'heating_supply_temperature_C', 'heating_return_temperature_C',
            'heating_storage_volume_flow_L_h', 'heating_mixer_fraction', 'storage_loss_W',
            'storage_balance_error_Wh', 'wp_flow_limited', 'heat_pump_commanded_power_fraction',
            'heat_pump_start_count', 'heat_pump_runtime_fraction', 'heat_pump_requested_thermal_power_W',
            'heat_pump_building_loss_feedforward_W', 'heat_pump_buffer_temperature_correction_W',
            'heat_pump_shutdown_check_heat_W', 'heat_pump_minimum_thermal_power_W',
            'heat_pump_relative_thermal_power')})
        hydraulic_rows[-1].update({key: getattr(result, key) for key in (
            'building_air_loss_W', 'building_mass_loss_W', 'building_energy_change_Wh',
            'building_balance_error_Wh', 'building_temperature_cap_loss_W',
            'heating_requested_power_W', 'hk_shift_K')})
        hydraulic_rows[-1]['mass_temperature_C'] = state.building.T_mass_C
        hydraulic_rows[-1]['storage_energy_Wh'] = params.storage.model.energy_Wh(state.storage.node_temperatures_C)
        hydraulic_rows[-1]['building_energy_Wh'] = params.building.energy_Wh(state.building)
        hydraulic_rows[-1]['heating_supply_target_C'] = result.heating_supply_target_C
        hydraulic_rows[-1]['heating_return_target_C'] = result.heating_return_target_C
        hydraulic_rows[-1]['heating_curve_reference'] = params.heating_curve_reference
        hydraulic_rows[-1]['storage_control_sensor_layer'] = params.storage.control_sensor_node + 1
        hydraulic_rows[-1]['wp_supply_storage_layer'] = params.storage.wp_supply_node + 1
        hydraulic_rows[-1]['wp_return_storage_layer'] = params.storage.wp_return_node + 1
        hydraulic_rows[-1]['heating_supply_storage_layer'] = params.storage.heating_supply_node + 1
        hydraulic_rows[-1]['heating_return_storage_layer'] = params.storage.heating_return_node + 1
        heat_pump_on.append(state.heat_pump.is_on)
        cop_values.append(result.cop)
        potential_cops.append(result.potential_cop)
        thermal_power_values.append(result.thermal_power_W)
        electric_power_values.append(result.electric_power_W)
        power_fractions.append(result.power_fraction)
        building_heating.append(result.heating_power_to_building_W)
        building_losses.append(result.building_heat_loss_W)
        reference_cops.append(result.reference_cop)
        cop_advantages.append(result.cop_advantage)
        fallback_active.append(getattr(state.heat_pump, 'fallback_running', False) and result.thermal_power_W > 0)
        fallback_override.append(getattr(state.heat_pump, 'fallback_override', False) and result.thermal_power_W > 0)

    return SimulationResult(
        settling_history=settling_history,
        measurement_initial_state=measurement_initial_state,
        initial_building_energy_Wh=params.building.energy_Wh(initial_building),
        initial_storage_energy_Wh=initial_storage_energy,
        settling_hours=settling_count * params.dt_hours,
        mass_temperatures_C=[r["mass_temperature_C"] for r in hydraulic_rows],
        building_parameters=params.building.derived_parameters(),
        initial_building_temperatures_C={"air_C": initial_building.T_room_C, "mass_C": initial_building.T_mass_C},
        timestamps=timestamps,
        outside_temperatures_C=outside_temperatures_C,
        room_temperatures_C=room_temperatures_C,
        storage_temperatures_C=storage_temperatures_C,
        heat_pump_on=heat_pump_on,
        cop=cop_values,
        potential_cop=potential_cops,
        thermal_power_W=thermal_power_values,
        electric_power_W=electric_power_values,
        total_electric_energy_Wh=segment_electric_Wh,
        total_thermal_energy_Wh=segment_thermal_Wh,
        night_setback=params.operating.night_setback,
        target_room_temperature_C=params.target_room_temperature_C,
        comfort_buffer_K=float(config['building']['comfort_buffer_K']),
        heat_pump_power_fractions=power_fractions,
        heat_pump_relative_thermal_powers=[r['heat_pump_relative_thermal_power'] for r in hydraulic_rows],
        heating_power_to_building_W=building_heating,
        building_heat_loss_W=building_losses,
        reference_cop=reference_cops,
        cop_advantage=cop_advantages,
        fallback_active=fallback_active,
        fallback_override=fallback_override,
        load_shifting_summary=_merge_summaries(summaries) if summaries else {},
        segment_starts=segment_starts,
        storage_energy_change_Wh=sum(storage_changes) if len(segments) > 1 else None,
        storage_node_temperatures_C=storage_nodes,
        hydraulics=hydraulic_rows,
        heat_pump_starts=starts,
        heat_pump_runtime_fractions=runtime_fractions,
        storage_mean_temperatures_C=storage_means,
    )


def output_interval_steps(config: dict[str, Any]) -> int:
    """Return how many physical steps separate recorded CSV rows."""
    dt = float(config['simulation']['dt_minutes'])
    if not isfinite(dt) or dt <= 0:
        raise ValueError('simulation.dt_minutes must be finite and positive')
    interval = float(config['simulation'].get('output_interval_minutes', dt))
    if not isfinite(interval) or interval <= 0:
        raise ValueError('simulation.output_interval_minutes must be finite and positive')
    steps = interval / dt
    if steps < 1 - 1e-8 or not isclose(steps, round(steps), abs_tol=1e-8):
        raise ValueError('simulation.output_interval_minutes must be an integer multiple of dt_minutes')
    return round(steps)


def build_result_rows(result: SimulationResult, output_steps: int = 1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(output_steps, bool) or not isinstance(output_steps, int) or output_steps < 1:
        raise ValueError('output_steps must be a positive integer')
    for idx in range(0, len(result.timestamps), output_steps):
        timestamp = result.timestamps[idx]
        rows.append(
            {
                "timestamp": timestamp,
                "outside_temperature_C": result.outside_temperatures_C[idx],
                "room_temperature_C": result.room_temperatures_C[idx],
                "storage_temperature_C": result.storage_temperatures_C[idx],
                "heat_pump_on": result.heat_pump_on[idx],
                "heat_pump_power_fraction": result.heat_pump_power_fractions[idx],
                "target_room_temperature_C": result.target_room_temperature_C - result.night_setback.room_reduction_K(datetime.fromisoformat(timestamp)),
                "comfort_threshold_C": result.target_room_temperature_C - result.comfort_buffer_K - result.night_setback.room_reduction_K(datetime.fromisoformat(timestamp), comfort=True),
                "cop": result.cop[idx],
                "potential_cop": result.potential_cop[idx],
                "reference_cop": result.reference_cop[idx],
                "cop_advantage": result.cop_advantage[idx],
                "fallback_active": result.fallback_active[idx],
                "fallback_override": result.fallback_override[idx],
                "thermal_power_W": result.thermal_power_W[idx],
                "heating_power_to_building_W": result.heating_power_to_building_W[idx],
                "building_heat_loss_W": result.building_heat_loss_W[idx],
                "electric_power_W": result.electric_power_W[idx],
            }
        )
        if result.hydraulics:
            rows[-1].update(result.hydraulics[idx])
        if result.storage_mean_temperatures_C:
            rows[-1]['storage_mean_temperature_C'] = result.storage_mean_temperatures_C[idx]
        if result.storage_node_temperatures_C:
            nodes = result.storage_node_temperatures_C[idx]
            rows[-1]['storage_control_temperature_C'] = result.storage_temperatures_C[idx]
            rows[-1].update(storage_top_temperature_C=nodes[0], storage_bottom_temperature_C=nodes[-1])
            rows[-1].update({f'storage_node_{n + 1}_temperature_C': t for n, t in enumerate(nodes)})
    return rows


def save_settling_csv(result: SimulationResult, path: Path, output_steps: int = 1):
    """Keep settling temperatures separate from all measured results."""
    import csv
    columns = list(result.settling_history[0]) if result.settling_history else [
        'timestamp', 'elapsed_hours', 'outside_temperature_C', 'air_temperature_C',
        'mass_temperature_C', 'storage_mean_temperature_C',
        'storage_top_temperature_C', 'storage_bottom_temperature_C']
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result.settling_history[::output_steps])
