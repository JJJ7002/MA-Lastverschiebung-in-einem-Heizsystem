from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from math import isfinite, isclose
from datetime import datetime, timedelta
from control.cop_optimized import CopOptimizedParams, update_cop_control
from control.heating_curve import (heating_curve_power_request, heating_curve_target,
                                   heating_supply_curve_power_request)
from control.temperature_targets import HeatingCurveParams

from control.operating_constraints import OperatingParams, OperatingState
from models.building import BuildingParams, BuildingState, advance_building
from models.heat_pump import HeatPumpParams, get_performance, validate_power_command
from models.storage import StorageParams, StorageState
from models.radiator import RadiatorMap
from models.hydraulics import ideal_heating, fixed_spread_heat_pump, four_ports
from control.rl_power import apply_power_request


@dataclass
class SimulationParams:
    dt_hours: float
    step_minutes: float
    building: BuildingParams
    storage: StorageParams
    heat_pump: HeatPumpParams
    operating: OperatingParams
    k_radiator_W_per_K: float
    target_room_temperature_C: float
    room_response_minutes: float = 30.0
    room_target_lookahead_minutes: float = 0.0
    radiator: RadiatorMap | None = None
    cop_control: CopOptimizedParams | None = None
    control_interval_minutes: float | None = None
    heating_volume_flow_L_h: float = 1500.0
    heating_circuit_curve: HeatingCurveParams | None = None
    heating_curve_reference: str = 'supply'
    heating_curve_control: bool = False
    heating_supply_curve_control: bool = False
    cop_fallback_supply_curve: bool = True  # cop_optimized fallback: heating_curve_supply (else heating_curve)
    heating_supply_curve_offset_K: float = 0.0
    heating_supply_curve_slope_factor: float = 1.0
    heating_curve_temperature_gain_W_per_K: float = 2000.0
    heating_supply_curve_temperature_gain_W_per_K: float = 2000.0
    heating_curve_temperature_hysteresis_K: float = 2.0
    heating_supply_curve_temperature_hysteresis_K: float = 2.0
    # Measured effect of a heating-circuit set-point shift (K) on the buffer target (K per K) at the
    # given outside temperatures; linear in between, constant outside the range.
    hk_shift_outside_C: tuple = (0.0,)
    hk_shift_gain_K_per_K: tuple = (1.0,)

    def __post_init__(self):
        outside, gain = self.hk_shift_outside_C, self.hk_shift_gain_K_per_K
        if (len(outside) == 0 or len(outside) != len(gain)
                or not all(isfinite(v) for v in (*outside, *gain))
                or any(b <= a for a, b in zip(outside, outside[1:]))
                or any(g <= 0 for g in gain)):
            raise ValueError('control.heating_curve.hk_shift_gain needs equally long, finite lists '
                             'outside_C (strictly increasing) and K_per_K (positive)')
        if not isfinite(self.room_response_minutes) or self.room_response_minutes <= 0:
            raise ValueError('heating.room_response_minutes must be finite and positive')
        if not isfinite(self.room_target_lookahead_minutes):
            raise ValueError('heating.room_target_lookahead_minutes must be finite')
        for name, value in (('heating_curve', self.heating_curve_temperature_hysteresis_K),
                            ('heating_curve_supply', self.heating_supply_curve_temperature_hysteresis_K)):
            if not isfinite(value) or value < 0:
                raise ValueError(f'control.{name}.temperature_hysteresis_K must be finite and nonnegative')
        for name, value in (('heating_curve', self.heating_curve_temperature_gain_W_per_K),
                            ('heating_curve_supply', self.heating_supply_curve_temperature_gain_W_per_K)):
            if not isfinite(value) or value < 0:
                raise ValueError(f'control.{name}.temperature_gain_W_per_K must be finite and nonnegative')
        if not isfinite(self.heating_supply_curve_slope_factor) or self.heating_supply_curve_slope_factor <= 0:
            raise ValueError('control.heating_curve_supply.slope_factor must be finite and positive')
        if not isfinite(self.heating_supply_curve_offset_K):
            raise ValueError('heating_supply_curve.offset_K must be finite')
        if self.heating_curve_reference not in ('supply', 'return'):
            raise ValueError('control.heating_circuit_curve.reference must be supply or return')
        if self.radiator is None:
            ideal_heating(21, 21, 0, self.k_radiator_W_per_K, self.heating_volume_flow_L_h)
        self.storage.model  # Validate finite-volume geometry before the run.
        if not isfinite(self.dt_hours) or self.dt_hours <= 0:
            raise ValueError('simulation.dt_minutes must be finite and positive')
        interval = self.dt_hours * 60 if self.control_interval_minutes is None else self.control_interval_minutes
        ratio = interval / (self.dt_hours * 60)
        if not isfinite(ratio) or ratio < 1 - 1e-8 or not isclose(ratio, round(ratio), abs_tol=1e-8):
            raise ValueError('control.interval_minutes must be a positive integer multiple of the simulation step')

    def hk_shift_gain(self, outside_C):
        return float(np.interp(outside_C, self.hk_shift_outside_C, self.hk_shift_gain_K_per_K))

    @property
    def control_steps(self) -> int:
        if self.control_interval_minutes is None:
            return 1
        return round(self.control_interval_minutes / (self.dt_hours * 60))

    def heating_curve_target_C(self, outside_C, timestamp=None):
        curve = self.heating_circuit_curve or self.operating.heating_curve
        return (curve.target_temperature_C(outside_C)
                - self.operating.night_setback.reduction_K(timestamp))

    def room_target_C_at(self, timestamp=None):
        return (self.target_room_temperature_C
                - self.operating.night_setback.room_reduction_K(timestamp))

    def heating_flow(self, hot_C, room_C, demand_W, outside_C, timestamp=None):
        if self.radiator is not None:
            return self.radiator.flow(hot_C, room_C, demand_W)
        target = self.heating_curve_target_C(outside_C, timestamp)
        targets = ({'return_target_C': target} if self.heating_curve_reference == 'return'
                   else {'supply_target_C': target})
        return ideal_heating(hot_C, room_C, demand_W, self.k_radiator_W_per_K,
                             self.heating_volume_flow_L_h, **targets)


    def room_heat_demand_W(self, room_C, outside_C, target_C, mass_C=None):
        return self.building.heat_request(BuildingState(room_C, mass_C), outside_C,
                                          target_C, self.room_response_minutes)


@dataclass
class SimulationState:
    building: BuildingState
    storage: StorageState
    heat_pump: OperatingState
    total_electric_energy_Wh: float = 0.0
    total_thermal_energy_Wh: float = 0.0
    power_fraction: float = -1.0
    commanded_power_fraction: float | None = None
    steps_elapsed: int = 0
    reference_cop: float | None = None
    cop_advantage: float | None = None
    buffer_target_C: float | None = None
    requested_thermal_power_W: float | None = None
    building_loss_feedforward_W: float | None = None
    buffer_temperature_correction_W: float | None = None
    shutdown_check_heat_W: float | None = None
    minimum_heat_W: float | None = None
    building_heat_W: float = 0.0
    hk_shift_K: float = 0.0  # Heating-circuit set-point shift chosen by an offset controller


@dataclass(frozen=True)
class SimulationInput:
    outside_temperature_C: float
    timestamp: datetime | None = None


@dataclass(frozen=True)
class StepResult:
    state: SimulationState
    cop: float
    thermal_power_W: float
    electric_power_W: float
    heating_power_to_building_W: float
    power_fraction: float
    building_heat_loss_W: float
    reference_cop: float | None = None
    cop_advantage: float | None = None
    potential_cop: float = 0.0
    buffer_target_C: float = 0.0
    wp_supply_temperature_C: float | None = None
    wp_return_temperature_C: float | None = None
    wp_volume_flow_L_h: float = 0.0
    wp_active_volume_flow_L_h: float = 0.0
    heating_supply_temperature_C: float = 0.0
    heating_return_temperature_C: float = 0.0
    heating_storage_volume_flow_L_h: float = 0.0
    heating_mixer_fraction: float = 0.0
    storage_loss_W: float = 0.0
    storage_balance_error_Wh: float = 0.0
    wp_flow_limited: bool = False
    heat_pump_commanded_power_fraction: float = -1.0
    heat_pump_start_count: int = 0
    heat_pump_runtime_fraction: float = 0.0
    heating_supply_target_C: float = 0.0
    heating_return_target_C: float | None = None
    wp_supply_target_temperature_C: float = 0.0
    heat_pump_requested_thermal_power_W: float | None = None
    heat_pump_building_loss_feedforward_W: float | None = None
    heat_pump_buffer_temperature_correction_W: float | None = None
    heat_pump_shutdown_check_heat_W: float | None = None
    heat_pump_minimum_thermal_power_W: float | None = None
    heat_pump_relative_thermal_power: float = 0.0
    building_air_loss_W: float = 0.0
    building_mass_loss_W: float = 0.0
    building_energy_change_Wh: float = 0.0
    building_balance_error_Wh: float = 0.0
    building_temperature_cap_loss_W: float = 0.0
    heating_requested_power_W: float = 0.0
    hk_shift_K: float = 0.0


def simulate_step(
    state: SimulationState,
    inputs: SimulationInput,
    params: SimulationParams,
    requested_power_fraction: float | None = None,
) -> StepResult:
    if requested_power_fraction is not None:
        validate_power_command(requested_power_fraction)
    # The configuration is authoritative, including manually supplied states.
    state = replace(state, storage=replace(state.storage,
                    control_sensor_layer=params.storage.control_sensor_node + 1))
    control_due = state.steps_elapsed % params.control_steps == 0
    buffer_target = state.buffer_target_C
    map_supply_target = min(state.storage.bottom_C + params.heat_pump.supply_spread_K,
                            params.heat_pump.sink_temp_max_C)
    requested_heat_W = state.requested_thermal_power_W if params.heating_curve_control else None
    loss_feedforward = state.building_loss_feedforward_W if params.heating_curve_control else None
    temperature_correction = state.buffer_temperature_correction_W if params.heating_curve_control else None
    shutdown_check = state.shutdown_check_heat_W if params.heating_curve_control else None
    minimum_heat = state.minimum_heat_W if params.heating_curve_control else None
    if control_due or buffer_target is None:
        if params.heating_curve_control:
            power_request, buffer_target = (
                heating_supply_curve_power_request(state, inputs, params)
                if params.heating_supply_curve_control
                else heating_curve_power_request(state, inputs, params))
            requested_heat_W = power_request.requested_heat_W
            loss_feedforward = power_request.building_loss_W
            temperature_correction = power_request.temperature_correction_W
            shutdown_check = power_request.shutdown_check_heat_W
            minimum_heat = power_request.minimum_heat_W
            if requested_power_fraction is None:
                requested_power_fraction = power_request.power_fraction
            else:
                requested_heat_W = None  # Explicit external fraction overrides the power controller.
        else:
            buffer_target = heating_curve_target(inputs.outside_temperature_C, inputs.timestamp, params,
                                                 state.hk_shift_K)
        buffer_target = min(buffer_target, params.heat_pump.sink_temp_max_C)
    reference_cop, cop_advantage = state.reference_cop, state.cop_advantage
    if not control_due:
        new_hp_state = replace(state.heat_pump,
                               minutes_in_current_state=state.heat_pump.minutes_in_current_state + params.step_minutes)
        power_fraction = (state.power_fraction if state.commanded_power_fraction is None
                          else state.commanded_power_fraction)
    elif requested_power_fraction is not None:
        new_hp_state, power_fraction = apply_power_request(
            requested_power_fraction, state.storage.control_C, state.heat_pump,
            params.operating, params.heat_pump,
            params.step_minutes, inputs.timestamp)
    elif params.cop_control is not None:
        if inputs.timestamp is None:
            raise ValueError('COP control requires a timestamp')
        new_hp_state, power_fraction, reference_cop, cop_advantage = update_cop_control(
            state.storage.control_C, inputs.outside_temperature_C, inputs.timestamp,
            state.heat_pump, params.operating, params.heat_pump, params.cop_control,
            params.step_minutes, fallback_power=(
                heating_supply_curve_power_request(state, inputs, params)
                if params.cop_fallback_supply_curve
                else heating_curve_power_request(state, inputs, params))[0].power_fraction,
            supply_target_C=map_supply_target)
    else:
        raise ValueError('A modulation command is required at each control decision')

    # A seasonal shutdown is a hard plant-level stop, even between regular
    # controller decisions or when another strategy supplies a power command.
    if params.operating.schedule.heating_off_at(inputs.timestamp):
        new_hp_state = replace(new_hp_state, is_on=False, minutes_in_current_state=0)
        power_fraction = -1.0

    potential_cop, _, _ = get_performance(
        source_temp_C=inputs.outside_temperature_C,
        sink_temp_C=map_supply_target,
        params=params.heat_pump,
    )
    if new_hp_state.is_on:
        commanded_power_fraction = power_fraction
    else:
        commanded_power_fraction = -1.0
    applied_command = commanded_power_fraction

    heating_return_target = (params.heating_curve_target_C(inputs.outside_temperature_C, inputs.timestamp)
                             if params.heating_curve_reference == 'return' else None)
    tank = params.storage.model
    temperatures = np.asarray(state.storage.node_temperatures_C
                              if state.storage.node_temperatures_C is not None
                              else [state.storage.temperature_C] * params.storage.node_count, dtype=float)
    if temperatures.shape != (params.storage.node_count,):
        raise ValueError('Storage state must match storage.node_count')
    temperatures = tank.mix_inversions(temperatures)
    building = state.building
    duration = params.dt_hours * 3600
    elapsed = 0.0
    totals = dict(heat=0.0, electric=0.0, heating=0.0, building_loss=0.0,
                  storage_loss=0.0, wp_flow=0.0, wp_supply=0.0, wp_return=0.0,
                  wp_active_seconds=0.0, hk_supply=0.0, hk_return=0.0,
                  hk_flow=0.0, mixer=0.0, balance=0.0, hk_target=0.0, wp_fraction=0.0,
                  wp_target=0.0, wp_relative_heat=0.0)
    totals.update(air_loss=0., mass_loss=0., building_energy=0., building_balance=0.,
                  building_temperature_cap_loss=0., demand=0.)
    flow_limited = False
    while duration - elapsed > 1e-8:
        remaining = duration - elapsed
        step_timestamp = (inputs.timestamp + timedelta(seconds=elapsed)
                          if inputs.timestamp is not None else None)
        # Physical room feedback uses its own time constant, not output duration.
        response_end = (inputs.timestamp + timedelta(seconds=elapsed
                                                     + params.room_target_lookahead_minutes * 60)
                        if inputs.timestamp is not None else None)
        response_target_C = params.room_target_C_at(response_end)
        demand = params.room_heat_demand_W(building.T_room_C,
                                           inputs.outside_temperature_C, response_target_C, building.T_mass_C)
        if params.operating.schedule.heating_off_at(step_timestamp):
            demand = 0.0
        heating = params.heating_flow(float(temperatures[params.storage.heating_supply_node]), building.T_room_C, demand,
                                      inputs.outside_temperature_C, inputs.timestamp)
        wp_return = float(temperatures[params.storage.wp_return_node])
        hp = fixed_spread_heat_pump(wp_return, inputs.outside_temperature_C,
                                 commanded_power_fraction, params.heat_pump)
        flow_limited |= hp.limited
        if hp.heat_W == 0 and commanded_power_fraction >= 0:
            # A nonpositive lift cannot deliver heat even during minimum run time.
            new_hp_state = replace(new_hp_state, is_on=False, minutes_in_current_state=0)
            commanded_power_fraction = -1.0
        ports = four_ports(params.storage.node_count, hp, heating)
        dt, advanced = tank.limited_step_fast(
            temperatures, ports, params.target_room_temperature_C, remaining)
        temperatures = advanced.temperatures_C
        building_step = advance_building(building, inputs.outside_temperature_C, heating.heat_W,
                                         params.building, dt / 3600)
        building = building_step.state
        building_loss = building_step.air_loss_W + building_step.mass_loss_W
        totals['air_loss'] += building_step.air_loss_W * dt
        totals['mass_loss'] += building_step.mass_loss_W * dt
        totals['building_temperature_cap_loss'] += building_step.temperature_cap_loss_W * dt
        totals['building_energy'] += building_step.energy_change_Wh
        totals['building_balance'] += building_step.balance_error_Wh
        totals['demand'] += demand * dt
        for key, value in (('heat', hp.heat_W), ('electric', hp.electricity_W),
                           ('heating', heating.heat_W), ('building_loss', building_loss),
                           ('storage_loss', advanced.loss_W), ('wp_flow', hp.mass_kg_s * 3600),
                           ('hk_supply', heating.supply_C), ('hk_return', heating.return_C),
                           ('hk_flow', heating.storage_kg_s * 3600), ('mixer', heating.mixer_fraction),
                           ('hk_target', heating.target_supply_C)):
            totals[key] += value * dt
        if hp.mass_kg_s > 0:
            totals['wp_relative_heat'] += hp.heat_W / get_performance(
                inputs.outside_temperature_C, hp.supply_C, params.heat_pump, 1)[1] * dt
            totals['wp_fraction'] += hp.fraction * dt
            totals['wp_target'] += (wp_return + params.heat_pump.supply_spread_K) * dt
            totals['wp_active_seconds'] += dt
            totals['wp_supply'] += hp.supply_C * dt
            totals['wp_return'] += hp.return_C * dt
        totals['balance'] += advanced.balance_error_Wh
        elapsed += dt

    thermal_power_W = totals['heat'] / duration
    electric_power_W = totals['electric'] / duration
    cop = thermal_power_W / electric_power_W if electric_power_W > 0 else 0.0
    power_fraction = (min(1.0, totals['wp_fraction'] / totals['wp_active_seconds'])
                      if totals['wp_active_seconds'] else -1.0)
    if abs(power_fraction - applied_command) < 1e-9:
        power_fraction = applied_command  # Remove only floating-point map/averaging noise.
    heating_power_to_building_W = totals['heating'] / duration
    new_storage = StorageState(float(temperatures[params.storage.control_sensor_node]),
                               tuple(float(t) for t in temperatures),
                               control_sensor_layer=params.storage.control_sensor_node + 1)

    new_state = SimulationState(
        building=building,
        storage=new_storage,
        heat_pump=new_hp_state,
        total_electric_energy_Wh=state.total_electric_energy_Wh + electric_power_W * params.dt_hours,
        total_thermal_energy_Wh=state.total_thermal_energy_Wh + thermal_power_W * params.dt_hours,
        power_fraction=power_fraction,
        steps_elapsed=state.steps_elapsed + 1,
        commanded_power_fraction=commanded_power_fraction,
        reference_cop=reference_cop,
        cop_advantage=cop_advantage,
        buffer_target_C=buffer_target,
        requested_thermal_power_W=requested_heat_W,
        building_loss_feedforward_W=loss_feedforward,
        buffer_temperature_correction_W=temperature_correction,
        shutdown_check_heat_W=shutdown_check,
        minimum_heat_W=minimum_heat,
        building_heat_W=heating_power_to_building_W,
        hk_shift_K=state.hk_shift_K,
    )

    return StepResult(
        state=new_state,
        cop=cop,
        thermal_power_W=thermal_power_W,
        electric_power_W=electric_power_W,
        heating_power_to_building_W=heating_power_to_building_W,
        power_fraction=power_fraction,
        reference_cop=reference_cop,
        cop_advantage=cop_advantage,
        potential_cop=potential_cop,
        building_heat_loss_W=totals['building_loss'] / duration,
        building_air_loss_W=totals['air_loss'] / duration,
        building_mass_loss_W=totals['mass_loss'] / duration,
        building_energy_change_Wh=totals['building_energy'],
        building_balance_error_Wh=totals['building_balance'],
        building_temperature_cap_loss_W=totals['building_temperature_cap_loss'] / duration,
        heating_requested_power_W=totals['demand'] / duration,
        buffer_target_C=buffer_target,
        wp_supply_temperature_C=(totals['wp_supply'] / totals['wp_active_seconds']
                                 if totals['wp_active_seconds'] else None),
        wp_return_temperature_C=(totals['wp_return'] / totals['wp_active_seconds']
                                 if totals['wp_active_seconds'] else None),
        wp_volume_flow_L_h=totals['wp_flow'] / duration,
        wp_active_volume_flow_L_h=(totals['wp_flow'] / totals['wp_active_seconds']
                                   if totals['wp_active_seconds'] else 0.0),
        heating_supply_temperature_C=totals['hk_supply'] / duration,
        heating_return_temperature_C=totals['hk_return'] / duration,
        heating_storage_volume_flow_L_h=totals['hk_flow'] / duration,
        heating_mixer_fraction=totals['mixer'] / duration,
        storage_loss_W=totals['storage_loss'] / duration,
        storage_balance_error_Wh=totals['balance'],
        wp_flow_limited=flow_limited,
        heat_pump_commanded_power_fraction=applied_command,
        heat_pump_start_count=int(not state.heat_pump.is_on and totals['wp_active_seconds'] > 0),
        heat_pump_runtime_fraction=totals['wp_active_seconds'] / duration,
        heating_supply_target_C=totals['hk_target'] / duration,
        heating_return_target_C=heating_return_target,
        wp_supply_target_temperature_C=(totals['wp_target'] / totals['wp_active_seconds']
            if totals['wp_active_seconds'] else float(temperatures[params.storage.wp_return_node])
            + params.heat_pump.supply_spread_K),
        heat_pump_requested_thermal_power_W=requested_heat_W,
        heat_pump_building_loss_feedforward_W=loss_feedforward,
        heat_pump_buffer_temperature_correction_W=temperature_correction,
        heat_pump_shutdown_check_heat_W=shutdown_check,
        heat_pump_minimum_thermal_power_W=minimum_heat,
        heat_pump_relative_thermal_power=totals['wp_relative_heat'] / duration,
        hk_shift_K=state.hk_shift_K,
    )
