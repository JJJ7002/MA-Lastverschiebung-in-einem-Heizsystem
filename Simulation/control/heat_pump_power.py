"""Shared target-to-power calculation, including the compressor power hysteresis."""
from dataclasses import dataclass
from math import isfinite

from models.heat_pump import heat_to_power_command
from models.hydraulics import fixed_spread_heat_pump


@dataclass(frozen=True)
class HeatPumpPowerRequest:
    requested_heat_W: float
    building_loss_W: float
    temperature_correction_W: float
    shutdown_check_heat_W: float
    minimum_heat_W: float
    power_fraction: float


def heat_pump_power_request(buffer_target_C, state, inputs, params,
                            temperature_hysteresis_K, temperature_gain_W_per_K,
                            *, building_loss_W=None):
    """Accept any strategy's buffer target and return a hysteresis-gated request.

    temperature_hysteresis_K and temperature_gain_W_per_K are supplied by the
    calling strategy (configured per strategy, e.g.
    control.heating_curve.temperature_hysteresis_K and .temperature_gain_W_per_K).

    Off: start strictly above minimum. On: stop strictly below minimum only
    when testing target + temperature hysteresis. The elevated target is used
    solely for shutdown; normal requested watts determine modulation. Command
    0 requests manufacturer minimum even at zero normal watts; -1 means off.
    Equipment runtime, schedule and safety constraints are applied downstream.
    """
    if not isfinite(buffer_target_C):
        raise ValueError('buffer_target_C must be finite')
    if not isfinite(temperature_hysteresis_K) or temperature_hysteresis_K < 0:
        raise ValueError('temperature_hysteresis_K must be finite and nonnegative')
    if not isfinite(temperature_gain_W_per_K) or temperature_gain_W_per_K < 0:
        raise ValueError('temperature_gain_W_per_K must be finite and nonnegative')
    loss = (sum(params.building.losses(state.building, inputs.outside_temperature_C))
            if building_loss_W is None else building_loss_W)
    correction = temperature_gain_W_per_K * (
        buffer_target_C - params.storage.control_temperature_C(state.storage))
    request = max(0.0, loss + correction)
    shutdown = max(0.0, loss + correction + temperature_gain_W_per_K
                   * temperature_hysteresis_K)
    # Evaluate manufacturer minimum at the actually achievable supply, including
    # volume-flow bounds. An infeasible operating point cannot start the machine.
    return_C = state.storage.bottom_C
    minimum = fixed_spread_heat_pump(return_C, inputs.outside_temperature_C, 0,
                                   params.heat_pump).heat_W
    run = minimum > 0 and (shutdown >= minimum if state.heat_pump.is_on else request > minimum)
    fraction = -1.0
    if run:
        fraction = (0.0 if request <= minimum else heat_to_power_command(
            request, inputs.outside_temperature_C,
            min(return_C + params.heat_pump.supply_spread_K, params.heat_pump.sink_temp_max_C),
            params.heat_pump))
    return HeatPumpPowerRequest(request, loss, correction, shutdown, minimum, fraction)
