"""Ideal mixing loop and SBP four-port layout; node 0 is the top."""
from dataclasses import dataclass
from math import isfinite

from models.heat_pump import available_power_fractions, get_performance, validate_power_command
from models.stratified_tank import CP_J_KG_K, PortFlow


@dataclass(frozen=True)
class HeatingFlow:
    supply_C: float
    return_C: float
    heat_W: float
    circulation_kg_s: float
    storage_kg_s: float
    mixer_fraction: float
    target_supply_C: float
    target_return_C: float | None = None


def ideal_heating(hot_C, room_C, demand_W, k_W_K, volume_flow_L_h, *,
                  supply_target_C=None, return_target_C=None):
    mass = volume_flow_L_h / 3600
    capacity_rate = mass * CP_J_KG_K
    if not isfinite(k_W_K) or k_W_K <= 0 or not isfinite(mass) or mass <= 0:
        raise ValueError('heating.k_radiator_W_per_K and volume_flow_L_h must be positive and finite')
    if capacity_rate < k_W_K:
        raise ValueError('Heating circulation must satisfy mass * cp >= radiator coefficient')
    demand_W = max(0.0, demand_W)
    if supply_target_C is not None and return_target_C is not None:
        raise ValueError('Choose either heating supply or return target')
    if return_target_C is not None:
        if not isfinite(return_target_C):
            raise ValueError('Heating return target must be finite')
        # Solve T_supply = T_return_target + Q / (mass * cp), with
        # Q = min(demand, k * max(T_supply - T_room, 0)). Room valves may
        # limit Q, so a fixed design temperature difference is insufficient.
        target = return_target_C
        if return_target_C > room_C:
            target += demand_W / capacity_rate
            if capacity_rate > k_W_K and demand_W > k_W_K * (target - room_C):
                target = room_C + (return_target_C - room_C) / (1 - k_W_K / capacity_rate)
        supply = min(target, hot_C)
    elif supply_target_C is None:
        target = room_C + demand_W / k_W_K
        supply = min(target, hot_C) if hot_C > room_C else room_C
    else:
        if not isfinite(supply_target_C):
            raise ValueError('Heating supply target must be finite')
        target = supply_target_C
        supply = min(target, hot_C)
    # Ideal room valves reduce heat below radiator capacity while the mixer
    # regulates the selected curve target through the required supply.
    heat = min(demand_W, max(0.0, k_W_K * (supply - room_C)))
    return_C = supply - heat / capacity_rate
    fraction = (supply - return_C) / (hot_C - return_C) if heat > 0 else 0.0
    return HeatingFlow(supply, return_C, heat, mass, mass * fraction, fraction,
                       target, return_target_C)


@dataclass(frozen=True)
class HeatPumpFlow:
    supply_C: float | None
    return_C: float
    mass_kg_s: float
    heat_W: float
    electricity_W: float
    cop: float
    fraction: float
    limited: bool = False




def fixed_spread_heat_pump(return_C, outside_C, request, params):
    """Track return + fixed water-side spread; solve achieved supply at pump bounds.

    Maximum flow allows increased spread, minimum flow decreases it. The absolute
    supply ceiling can reduce compressor output or stop it below map minimum.
    """
    validate_power_command(request)
    def performance(supply):
        return get_performance(outside_C, supply, params, request)
    target = min(return_C + params.supply_spread_K, params.sink_temp_max_C)
    spread = target - return_C
    if request == -1 or spread <= params.minimum_supply_spread_K:
        return HeatPumpFlow(None, return_C, 0, 0, 0, 0, -1, request >= 0)
    cop, heat, electricity = performance(target)
    mass = heat / (CP_J_KG_K * spread)
    minimum = params.min_volume_flow_L_h / 3600
    maximum = (params.max_volume_flow_L_h / 3600
               if params.max_volume_flow_L_h is not None else float('inf'))
    if minimum <= mass <= maximum:
        return HeatPumpFlow(target, return_C, mass, heat, electricity, cop, request,
                            target < return_C + params.supply_spread_K)
    mass = min(maximum, max(minimum, mass))
    capacity_rate = mass * CP_J_KG_K
    low, high = return_C, params.sink_temp_max_C
    low_error = -performance(low)[1] / capacity_rate
    high_error = high - return_C - performance(high)[1] / capacity_rate
    if high_error < 0:
        # At maximum supply temperature the requested heat exceeds water uptake.
        allowed = capacity_rate * (high - return_C)
        minimum_heat = get_performance(outside_C, high, params, 0)[1]
        if allowed < minimum_heat - 1e-8:
            return HeatPumpFlow(None, return_C, 0, 0, 0, 0, -1, True)
        # Largest manufacturer map level the water can still carry at the ceiling
        # (rounding up would deliver more heat than mass * cp * (supply - return)).
        actual_command = max(level for level in available_power_fractions(params)
                             if get_performance(outside_C, high, params, level)[1] <= allowed + 1e-9)
        cop, heat, electricity = get_performance(outside_C, high, params, actual_command)
        mass = max(minimum, heat / (CP_J_KG_K * (high - return_C)))
        supply = return_C + heat / (mass * CP_J_KG_K)
        return HeatPumpFlow(supply, return_C, mass, heat, electricity, cop, actual_command, True)
    for iteration in range(80):
        trial = ((low + high) / 2 if iteration % 4 == 3 else
                 low - low_error * (high - low) / (high_error - low_error))
        error = trial - return_C - performance(trial)[1] / capacity_rate
        if abs(error) < 1e-9:
            break
        if error > 0:
            high, high_error = trial, error
        else:
            low, low_error = trial, error
    else:
        raise ArithmeticError('Fixed-flow WP supply/map solution did not converge')
    cop, heat, electricity = performance(trial)
    supply = return_C + heat / capacity_rate
    return HeatPumpFlow(supply, return_C, mass, heat, electricity, cop, request, True)


def four_ports(node_count, hp, heating):
    bottom = node_count - 1
    return (PortFlow(min(1, bottom), hp.mass_kg_s, hp.supply_C),
            PortFlow(bottom, -hp.mass_kg_s),
            PortFlow(0, -heating.storage_kg_s),
            PortFlow(max(0, bottom - 1), heating.storage_kg_s, heating.return_C))
