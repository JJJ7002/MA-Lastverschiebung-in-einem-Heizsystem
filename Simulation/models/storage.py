from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import pi
from math import isfinite


WATER_HEAT_CAPACITY_J_per_kgK = 4182.0
WATER_HEAT_CAPACITY_Wh_per_kgK = WATER_HEAT_CAPACITY_J_per_kgK / 3600.0
WATER_DENSITY_kg_per_L = 1.0


def resolve_sensor_node(node_count: int, layer: int | None) -> int:
    """1-based layers from the top; None follows the second-bottom layer."""
    if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count < 1:
        raise ValueError('storage.node_count must be a positive integer')
    if layer is None:
        return max(0, node_count - 2)
    if isinstance(layer, bool) or not isinstance(layer, int) or not 1 <= layer <= node_count:
        raise ValueError('storage.control_sensor_layer must be an integer from 1 to node_count, or null')
    return layer - 1


@dataclass(frozen=True)
class StorageParams:
    volume_L: float
    ua_W_per_K: float
    node_count: int = 10
    diameter_m: float = 0.630
    control_sensor_layer: int | None = None

    def __post_init__(self):
        resolve_sensor_node(self.node_count, self.control_sensor_layer)

    @cached_property
    def model(self):
        from models.stratified_tank import StorageParameters, StratifiedStorage
        return StratifiedStorage(StorageParameters(
            volume_L=self.volume_L, ua_W_K=self.ua_W_per_K,
            node_count=self.node_count, diameter_m=self.diameter_m))

    @property
    def water_height_m(self) -> float:
        return self.volume_L / 1000 / (pi * self.diameter_m**2 / 4)

    @property
    def mass_kg(self) -> float:
        return self.volume_L * WATER_DENSITY_kg_per_L

    @property
    def heat_capacity_Wh_per_K(self) -> float:
        return self.mass_kg * WATER_HEAT_CAPACITY_Wh_per_kgK

    @property
    def top_heat_capacity_Wh_per_K(self) -> float:
        return float(self.model.capacities_J_K[0] / 3600)

    @property
    def control_heat_capacity_Wh_per_K(self) -> float:
        return float(self.model.capacities_J_K[self.control_sensor_node] / 3600)

    @property
    def wp_supply_node(self) -> int:
        return min(1, self.node_count - 1)

    @property
    def wp_return_node(self) -> int:
        return self.node_count - 1

    @property
    def heating_supply_node(self) -> int:
        return 0

    @property
    def heating_return_node(self) -> int:
        return max(0, self.node_count - 2)

    @property
    def control_sensor_node(self) -> int:
        return resolve_sensor_node(self.node_count, self.control_sensor_layer)

    def control_temperature_C(self, state: StorageState) -> float:
        if state.node_temperatures_C is None:
            return state.temperature_C
        if len(state.node_temperatures_C) != self.node_count:
            raise ValueError('Storage state must match storage.node_count')
        return float(state.node_temperatures_C[self.control_sensor_node])

    def surplus_energy_Wh(self, state: StorageState, reference_C: float) -> float:
        """Heat above the target in each layer; cold layers do not cancel hot reserves."""
        if state.node_temperatures_C is None:
            return self.heat_capacity_Wh_per_K * max(0.0, state.temperature_C - reference_C)
        if len(state.node_temperatures_C) != self.node_count:
            raise ValueError('Storage state must match storage.node_count')
        return float(sum(capacity / 3600 * max(0.0, temperature - reference_C)
                         for capacity, temperature in zip(self.model.capacities_J_K, state.node_temperatures_C)))

    def heat_loss_W(self, state: StorageState, ambient_C: float) -> float:
        if state.node_temperatures_C is None:
            return self.ua_W_per_K * (state.mean_temperature_C - ambient_C)
        if len(state.node_temperatures_C) != self.node_count:
            raise ValueError('Storage state must match storage.node_count')
        return float(sum(ua * (temperature - ambient_C)
                         for ua, temperature in zip(self.model.ua_W_K, state.node_temperatures_C)))


@dataclass
class StorageState:
    """Selected buffer control temperature and full profile."""

    temperature_C: float
    node_temperatures_C: tuple[float, ...] | None = None
    control_sensor_layer: int | None = None

    def __post_init__(self):
        if self.control_sensor_layer is not None and (
                isinstance(self.control_sensor_layer, bool) or not isinstance(self.control_sensor_layer, int)
                or self.control_sensor_layer < 1):
            raise ValueError('storage.control_sensor_layer must be a positive integer or null')
        if self.node_temperatures_C is not None:
            self.node_temperatures_C = tuple(self.node_temperatures_C)
            if not self.node_temperatures_C or not all(isfinite(v) for v in self.node_temperatures_C):
                raise ValueError('Storage node temperatures must be finite and nonempty')
            self.temperature_C = self.control_C
        elif not isfinite(self.temperature_C):
            raise ValueError('Initial storage temperature must be finite')

    @property
    def control_C(self):
        if self.node_temperatures_C is None:
            return self.temperature_C
        return self.node_temperatures_C[resolve_sensor_node(len(self.node_temperatures_C), self.control_sensor_layer)]

    @property
    def top_C(self):
        return self.node_temperatures_C[0] if self.node_temperatures_C is not None else self.temperature_C

    @property
    def bottom_C(self):
        return self.node_temperatures_C[-1] if self.node_temperatures_C is not None else self.temperature_C

    @property
    def mean_temperature_C(self):
        if self.node_temperatures_C is None:
            return self.temperature_C
        return sum(self.node_temperatures_C) / len(self.node_temperatures_C)
