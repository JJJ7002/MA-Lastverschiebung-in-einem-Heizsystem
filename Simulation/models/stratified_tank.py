"""Explicit finite volumes with Kleinbach-style advection and inversion mixing.

Nodes are numbered from top to bottom. Time is in seconds, mass flow in kg/s.
    Axial conduction is an explicit model extension to Kleinbach (1990), section 2.1.
"""
from dataclasses import dataclass
from math import pi

import numpy as np

from models.storage import WATER_DENSITY_kg_per_L, WATER_HEAT_CAPACITY_Wh_per_kgK
from models.validation import number
from models import _tank_kernels

CP_J_KG_K = WATER_HEAT_CAPACITY_Wh_per_kgK * 3600.0


@dataclass(frozen=True)
class StorageParameters:
    volume_L: float
    node_count: int
    diameter_m: float
    ua_W_K: float
    courant_limit: float = 0.5
    node_volume_fractions: tuple[float, ...] | None = None

    def __post_init__(self):
        if isinstance(self.node_count, bool) or not isinstance(self.node_count, int) or self.node_count < 1:
            raise ValueError("storage.node_count must be a positive integer")
        for name in ("volume_L", "diameter_m"):
            number(getattr(self, name), f"storage.{name}", strictly_positive=True)
        number(self.ua_W_K, "storage.ua_W_K", minimum=0)
        number(self.courant_limit, "storage.courant_limit", strictly_positive=True, maximum=1)
        if self.node_volume_fractions is not None:
            if len(self.node_volume_fractions) != self.node_count:
                raise ValueError("node_volume_fractions must match storage.node_count")
            for value in self.node_volume_fractions:
                number(value, "node_volume_fraction", strictly_positive=True)
            if not np.isclose(sum(self.node_volume_fractions), 1.0, rtol=0, atol=1e-9):
                raise ValueError("node_volume_fractions must sum to 1")

    @property
    def fractions(self):
        if self.node_volume_fractions is None:
            return np.full(self.node_count, 1.0 / self.node_count)
        return np.asarray(self.node_volume_fractions, dtype=float)


@dataclass(frozen=True)
class PortFlow:
    node: int
    mass_flow_kg_s: float  # Positive into the tank, negative out.
    inlet_temperature_C: float | None = None


@dataclass(frozen=True)
class StorageStep:
    temperatures_C: np.ndarray
    port_heat_W: float
    loss_W: float
    energy_change_Wh: float
    balance_error_Wh: float


class StratifiedStorage:
    def __init__(self, params: StorageParameters):
        self.params = params
        self.water_height_m = params.volume_L / 1000 / (pi * params.diameter_m**2 / 4)
        # Fixed physical property: users configure geometry, not water material data.
        water_conductivity_W_mK = 0.6
        self.masses_kg = params.volume_L * WATER_DENSITY_kg_per_L * params.fractions
        self.capacities_J_K = self.masses_kg * CP_J_KG_K
        self.heights_m = self.water_height_m * params.fractions
        # Node boundaries and centres measured upwards from the water bottom.
        self.boundaries_m = self.water_height_m - np.r_[0.0, np.cumsum(self.heights_m)]
        self.centres_m = (self.boundaries_m[:-1] + self.boundaries_m[1:]) / 2
        # Split total UA over equivalent cylinder area (mantle + both ends).
        end_loss_fraction = params.diameter_m / (2*self.water_height_m + params.diameter_m)
        self.ua_W_K = params.ua_W_K * (1 - end_loss_fraction) * params.fractions
        self.ua_W_K[0] += params.ua_W_K * end_loss_fraction / 2
        self.ua_W_K[-1] += params.ua_W_K * end_loss_fraction / 2
        area = params.volume_L / 1000 / self.water_height_m
        distances = (self.heights_m[:-1] + self.heights_m[1:]) / 2
        self.conductances_W_K = water_conductivity_W_mK * area / distances

    def node_at_fraction(self, height_fraction):
        number(height_fraction, "storage port height_fraction", maximum=1)
        height = height_fraction * self.water_height_m
        return int(np.clip(np.searchsorted(-self.boundaries_m, -height, side="right") - 1,
                           0, self.params.node_count - 1))

    def sensor_temperature(self, temperatures_C, height_fraction):
        number(height_fraction, "storage sensor height_fraction", maximum=1)
        return float(np.interp(height_fraction * self.water_height_m,
                               self.centres_m[::-1], temperatures_C[::-1]))

    def mean_temperature(self, temperatures_C):
        return float(np.dot(self.params.fractions, temperatures_C))

    def energy_Wh(self, temperatures_C):
        """Sensible energy relative to 0 C, used only with a stated reference."""
        return float(np.dot(self.capacities_J_K, temperatures_C) / 3600)

    def _rates(self, temperatures_C, ports, ambient_C):
        temperatures = np.asarray(temperatures_C, dtype=float)
        if temperatures.shape != (self.params.node_count,) or not np.isfinite(temperatures).all():
            raise ValueError("Storage temperatures must be a finite node vector")
        number(ambient_C, "ambient_temperature_C", minimum=-273.15)
        net_mass = np.zeros(self.params.node_count)
        heat_W = np.zeros(self.params.node_count)
        outflow = np.zeros(self.params.node_count)
        for port in ports:
            if not 0 <= port.node < self.params.node_count:
                raise ValueError("Storage port node outside tank")
            flow = number(port.mass_flow_kg_s, "port.mass_flow_kg_s", minimum=-1e6)
            net_mass[port.node] += flow
            if flow > 0:
                if port.inlet_temperature_C is None:
                    raise ValueError("An entering port requires an inlet temperature")
                inlet = number(port.inlet_temperature_C, "port.inlet_temperature_C", minimum=-273.15)
                heat_W[port.node] += flow * CP_J_KG_K * inlet
            elif flow < 0:
                heat_W[port.node] += flow * CP_J_KG_K * temperatures[port.node]
                outflow[port.node] -= flow
        if abs(float(net_mass.sum())) > 1e-9:
            raise ValueError("Storage ports must conserve mass")
        port_heat = float(heat_W.sum())
        for i, flow in enumerate(np.cumsum(net_mass)[:-1]):
            donor = i if flow >= 0 else i + 1
            enthalpy = flow * CP_J_KG_K * temperatures[donor]
            heat_W[i] -= enthalpy
            heat_W[i + 1] += enthalpy
            outflow[donor] += abs(flow)
        conduction = self.conductances_W_K * (temperatures[:-1] - temperatures[1:])
        heat_W[:-1] -= conduction
        heat_W[1:] += conduction
        loss = self.ua_W_K * (temperatures - ambient_C)
        heat_W -= loss
        diagonal = outflow * CP_J_KG_K + self.ua_W_K
        diagonal[:-1] += self.conductances_W_K
        diagonal[1:] += self.conductances_W_K
        return heat_W, diagonal, port_heat, float(loss.sum())

    def max_step_seconds(self, temperatures_C, ports, ambient_C):
        _, diagonal, _, _ = self._rates(temperatures_C, ports, ambient_C)
        active = diagonal > 0
        if not active.any():
            return float("inf")
        return float(self.params.courant_limit * np.min(self.capacities_J_K[active] / diagonal[active]))

    def mix_inversions(self, temperatures_C):
        """Pool adjacent unstable blocks; preserve energy, including unequal nodes."""
        if _tank_kernels.KERNELS_AVAILABLE:
            return _tank_kernels.mix_inversions(
                np.ascontiguousarray(temperatures_C, dtype=np.float64), self.capacities_J_K)
        blocks = []
        for i, (temperature, capacity) in enumerate(zip(temperatures_C, self.capacities_J_K)):
            blocks.append([i, i + 1, capacity, capacity * temperature])
            while len(blocks) > 1 and blocks[-2][3] / blocks[-2][2] < blocks[-1][3] / blocks[-1][2]:
                lower = blocks.pop()
                upper = blocks.pop()
                blocks.append([upper[0], lower[1], upper[2] + lower[2], upper[3] + lower[3]])
        result = np.empty(self.params.node_count)
        for start, end, capacity, energy in blocks:
            result[start:end] = energy / capacity
        return result

    def step(self, temperatures_C, ports, ambient_C, dt_seconds):
        number(dt_seconds, "storage dt_seconds", strictly_positive=True)
        old = np.asarray(temperatures_C, dtype=float)
        rates, diagonal, port_heat, loss = self._rates(old, ports, ambient_C)
        # Keep every explicit finite-volume weight nonnegative. The simulation
        # loop chooses dt <= max_step_seconds; this guard protects direct callers.
        if np.any(dt_seconds * diagonal >
                  self.capacities_J_K * self.params.courant_limit * (1 + 1e-9)):
            raise ValueError("Storage step exceeds its conservative CFL limit")
        new = self.mix_inversions(
            old + rates * dt_seconds / self.capacities_J_K)
        change = float(np.dot(self.capacities_J_K, new - old) / 3600)
        residual = change - (port_heat - loss) * dt_seconds / 3600
        return StorageStep(new, port_heat, loss, change, residual)

    def limited_step(self, temperatures_C, ports, ambient_C, max_seconds):
        """Apply the CFL limit and advance, evaluating the heat fluxes once."""
        number(max_seconds, "storage dt_seconds", strictly_positive=True)
        old = np.asarray(temperatures_C, dtype=float)
        rates, diagonal, port_heat, loss = self._rates(old, ports, ambient_C)
        active = diagonal > 0
        limit = (float(self.params.courant_limit * np.min(self.capacities_J_K[active] / diagonal[active]))
                 if active.any() else float('inf'))
        dt_seconds = min(max_seconds, limit)
        number(dt_seconds, "storage dt_seconds", strictly_positive=True)
        new = self.mix_inversions(old + rates * dt_seconds / self.capacities_J_K)
        change = float(np.dot(self.capacities_J_K, new - old) / 3600)
        residual = change - (port_heat - loss) * dt_seconds / 3600
        return dt_seconds, StorageStep(new, port_heat, loss, change, residual)

    def limited_step_fast(self, temperatures_C, ports, ambient_C, max_seconds):
        """Same as limited_step; uses the numba kernel when available.

        Temperatures are bit-identical to limited_step. Diagnostic sums
        (port heat, loss, energy balance) may differ in the last digits.
        """
        if not _tank_kernels.KERNELS_AVAILABLE:
            return self.limited_step(temperatures_C, ports, ambient_C, max_seconds)
        number(max_seconds, "storage dt_seconds", strictly_positive=True)
        number(ambient_C, "ambient_temperature_C", minimum=-273.15)
        old = np.ascontiguousarray(temperatures_C, dtype=np.float64)
        if old.shape != (self.params.node_count,):
            raise ValueError("Storage temperatures must be a finite node vector")
        count = len(ports)
        nodes = np.empty(count, dtype=np.int64)
        flows = np.empty(count)
        inlets = np.full(count, np.nan)
        for i, port in enumerate(ports):
            nodes[i] = port.node
            flows[i] = port.mass_flow_kg_s
            if port.inlet_temperature_C is not None:
                inlets[i] = port.inlet_temperature_C
        if not np.isfinite(flows).all() or not np.isfinite(inlets[~np.isnan(inlets)]).all():
            raise ValueError("Storage port flows and inlet temperatures must be finite")
        new, dt, port_heat, loss, change, residual, error = _tank_kernels.tank_step(
            old, self.capacities_J_K, self.ua_W_K, self.conductances_W_K, nodes, flows, inlets,
            CP_J_KG_K, float(ambient_C), float(max_seconds), self.params.courant_limit)
        if error:
            raise ValueError({1: "Storage temperatures must be a finite node vector",
                              2: "Storage port node outside tank",
                              3: "Storage ports must conserve mass",
                              4: "An entering port requires an inlet temperature"}[error])
        if not dt > 0:
            raise ValueError("storage dt_seconds must be strictly positive")
        return dt, StorageStep(new, port_heat, loss, change, residual)
