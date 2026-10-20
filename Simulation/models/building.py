"""Energy-conserving two-node model of room air and effective building mass."""
from dataclasses import dataclass, asdict
from functools import cached_property
from math import isfinite


@dataclass(frozen=True)
class BuildingParams:
    UA_W_per_K: float
    heated_floor_area_m2: float = 300.0
    mass_heat_capacity_Wh_per_m2K: float = 72.2222222222
    mean_room_height_m: float = 2.5
    direct_loss_fraction: float = 0.5
    T_max_C: float = 25.0

    def __post_init__(self):
        try:
            valid_ua = isfinite(self.UA_W_per_K) and self.UA_W_per_K >= 0
        except (TypeError, ValueError):
            valid_ua = False
        if not valid_ua:
            raise ValueError('building.UA_W_per_K must be finite and nonnegative')
        if not isfinite(self.T_max_C):
            raise ValueError('building.T_max_C must be finite')
        for name in ('heated_floor_area_m2', 'mass_heat_capacity_Wh_per_m2K', 'mean_room_height_m'):
            self._positive(getattr(self, name), name)
        try:
            valid_fraction = isfinite(self.direct_loss_fraction) and 0 <= self.direct_loss_fraction <= 1
        except (TypeError, ValueError):
            valid_fraction = False
        if not valid_fraction:
            raise ValueError('building.direct_loss_fraction must be between 0 and 1')

    @staticmethod
    def _positive(value, name):
        try:
            valid = isfinite(value) and value > 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f'building.{name} must be finite and positive')

    @classmethod
    def from_config(cls, config):
        names = cls.__dataclass_fields__
        return cls(**{key: value for key, value in config.items() if key in names})

    @cached_property
    def C_air_Wh_per_K(self):
        volume = self.heated_floor_area_m2 * self.mean_room_height_m
        return 1200 * volume / 3600

    @cached_property
    def C_mass_Wh_per_K(self):
        return self.mass_heat_capacity_Wh_per_m2K * self.heated_floor_area_m2

    @cached_property
    def H_am_W_per_K(self):
        h_as = 3.45 * 4.5 * self.heated_floor_area_m2
        # The effective internal surface area retains the heavy-building
        # reference factor while the mass heat capacity is directly configurable.
        h_sm = 9.1 * 3.0 * self.heated_floor_area_m2
        return h_as*h_sm/(h_as+h_sm)

    @cached_property
    def H_ae_W_per_K(self):
        return self.direct_loss_fraction * self.UA_W_per_K

    @cached_property
    def H_me_W_per_K(self):
        # UA is split directly: air->outside gets direct_loss_fraction, mass->outside the rest.
        # Each loss uses its own node temperature, so air and mass losses differ when
        # the temperatures differ.
        return (1-self.direct_loss_fraction) * self.UA_W_per_K

    @cached_property
    def stationary_UA_W_per_K(self):
        """Steady-state air-to-outside conductance: direct path plus mass path in series."""
        indirect = self.H_am_W_per_K*self.H_me_W_per_K/(self.H_am_W_per_K+self.H_me_W_per_K)
        return self.H_ae_W_per_K + indirect

    def initialize(self, air_C, outside_C=None, mass_C=None):
        if mass_C is None:
            if outside_C is None:
                raise ValueError('Two-node initialization requires the first outside temperature or T_mass_init_C')
            mass_C = (self.H_am_W_per_K*air_C + self.H_me_W_per_K*outside_C)/(self.H_am_W_per_K+self.H_me_W_per_K)
        if not isfinite(air_C) or not isfinite(mass_C):
            raise ValueError('Initial building temperatures must be finite')
        return BuildingState(min(air_C, self.T_max_C), min(mass_C, self.T_max_C))

    def losses(self, state, outside_C):
        if state.T_mass_C is None:
            raise ValueError('Two-node building requires T_mass_C in its state')
        return (self.H_ae_W_per_K*(state.T_room_C-outside_C),
                self.H_me_W_per_K*(state.T_mass_C-outside_C))

    def heat_request(self, state, outside_C, target_C, response_minutes):
        direct, _ = self.losses(state, outside_C)
        coupling = self.H_am_W_per_K*(state.T_room_C-state.T_mass_C)
        return max(0., direct+coupling+self.C_air_Wh_per_K*(target_C-state.T_room_C)/(response_minutes/60))

    def energy_Wh(self, state):
        return self.C_air_Wh_per_K*state.T_room_C + self.C_mass_Wh_per_K*state.T_mass_C

    def derived_parameters(self):
        result = asdict(self)
        result.update({name: getattr(self, name) for name in
                       ('C_air_Wh_per_K', 'C_mass_Wh_per_K', 'H_am_W_per_K', 'H_ae_W_per_K', 'H_me_W_per_K',
                        'stationary_UA_W_per_K')})
        return result


@dataclass
class BuildingState:
    T_room_C: float
    T_mass_C: float


@dataclass(frozen=True)
class BuildingStep:
    state: BuildingState
    air_loss_W: float
    mass_loss_W: float
    temperature_cap_loss_W: float
    energy_change_Wh: float
    balance_error_Wh: float


def advance_building(state, T_out_C, Q_heat_W, params, dt_hours):
    if not isfinite(dt_hours) or dt_hours <= 0:
        raise ValueError('Building step duration must be finite and positive')
    params.losses(state, T_out_C)  # Check complete state.
    ca, cm = params.C_air_Wh_per_K/dt_hours, params.C_mass_Wh_per_K/dt_hours
    ham, hae, hme = params.H_am_W_per_K, params.H_ae_W_per_K, params.H_me_W_per_K
    # Backward Euler: all heat fluxes use the same end-of-step temperatures.
    a, d = ca+ham+hae, cm+ham+hme
    rhs_a, rhs_m = ca*state.T_room_C+Q_heat_W+hae*T_out_C, cm*state.T_mass_C+hme*T_out_C
    determinant = ca*cm+ca*(ham+hme)+cm*(ham+hae)+ham*hme+hae*ham+hae*hme
    unconstrained = BuildingState((d*rhs_a+ham*rhs_m)/determinant,
                                  (ham*rhs_a+a*rhs_m)/determinant)
    new = BuildingState(min(unconstrained.T_room_C, params.T_max_C),
                        min(unconstrained.T_mass_C, params.T_max_C))
    capped = (unconstrained.T_room_C > params.T_max_C
              or unconstrained.T_mass_C > params.T_max_C
              or state.T_room_C > params.T_max_C or state.T_mass_C > params.T_max_C)
    air_loss, mass_loss = params.losses(new, T_out_C)
    delta = params.energy_Wh(new)-params.energy_Wh(state)
    cap_loss = ((Q_heat_W-air_loss-mass_loss)-delta/dt_hours) if capped else 0.0
    error = delta-(Q_heat_W-air_loss-mass_loss-cap_loss)*dt_hours
    return BuildingStep(new, air_loss, mass_loss, cap_loss, delta, error)
