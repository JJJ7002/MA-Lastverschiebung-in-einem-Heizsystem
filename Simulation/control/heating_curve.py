"""Radiator-derived buffer target with shared modulating power control."""
from datetime import timedelta

from control.heat_pump_power import heat_pump_power_request


def heating_curve_target(outside_C, timestamp, params, hk_shift_K=0.0):
    """Buffer target of the radiator curve.

    hk_shift_K is a shift of the heating-circuit set-point of the heat pump controller (WPM "WP HK"):
    it moves the target by hk_shift_K times the measured curve gain at the current outside
    temperature (params.hk_shift_gain). Zero leaves the curve unchanged.
    """
    room_target = (params.target_room_temperature_C
                   - params.operating.night_setback.room_reduction_K(timestamp))
    target = params.operating.heating_curve.target_temperature_C(outside_C, room_target)
    if hk_shift_K:
        target += hk_shift_K * params.hk_shift_gain(outside_C)
    return min(params.heat_pump.sink_temp_max_C, target)


def heating_curve_power_request(state, inputs, params):
    target = heating_curve_target(inputs.outside_temperature_C, inputs.timestamp, params,
                                  state.hk_shift_K)
    room_target = params.room_target_C_at(inputs.timestamp)
    stationary_loss = max(0.0, params.building.stationary_UA_W_per_K
                          * (room_target - inputs.outside_temperature_C))
    return heat_pump_power_request(target, state, inputs, params,
                                   params.heating_curve_temperature_hysteresis_K,
                                   params.heating_curve_temperature_gain_W_per_K,
                                   building_loss_W=stationary_loss), target


def heating_supply_curve_target(outside_C, timestamp, state, params):
    """Derive the buffer target from the current required circuit supply.

    The required supply rise above the room target is scaled by
    heating_supply_curve_slope_factor (1 = unchanged), then the offset is added.
    """
    target_time = (timestamp + timedelta(minutes=params.room_target_lookahead_minutes)
                   if timestamp is not None else None)
    room_target = params.room_target_C_at(target_time)
    demand = params.room_heat_demand_W(
        state.building.T_room_C, outside_C, room_target, state.building.T_mass_C)
    if state.storage.node_temperatures_C is None:
        hot_C = state.storage.top_C
    else:
        hot_C = state.storage.node_temperatures_C[params.storage.heating_supply_node]
    flow = params.heating_flow(hot_C, state.building.T_room_C, demand, outside_C, timestamp)
    supply_C = room_target + params.heating_supply_curve_slope_factor * (flow.target_supply_C - room_target)
    return min(params.heat_pump.sink_temp_max_C,
               supply_C + params.heating_supply_curve_offset_K)


def heating_supply_curve_power_request(state, inputs, params):
    """Use required heating-circuit supply plus a fixed offset as buffer target."""
    target = heating_supply_curve_target(
        inputs.outside_temperature_C, inputs.timestamp, state, params)
    return heat_pump_power_request(target, state, inputs, params,
                                   params.heating_supply_curve_temperature_hysteresis_K,
                                   params.heating_supply_curve_temperature_gain_W_per_K), target
