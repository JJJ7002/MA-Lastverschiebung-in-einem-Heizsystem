"""Stationary equivalent radiator, tabulated in heat output (SI units)."""
from dataclasses import dataclass
import numpy as np
from models.stratified_tank import CP_J_KG_K


@dataclass
class RadiatorMap:
    reference_power_W: float = 6000.0
    reference_supply_C: float = 48.0
    reference_room_C: float = 22.0
    exponent: float = 1.3
    volume_flow_L_h: float = 1500.0
    max_power_W: float = 1000000.0
    power_step_W: float = 10.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not np.isfinite(value):
                raise ValueError(f'heating.radiator.{name} must be finite')
        for name in ('reference_power_W', 'exponent', 'volume_flow_L_h', 'max_power_W', 'power_step_W'):
            if getattr(self, name) <= 0:
                raise ValueError(f'heating.radiator.{name} must be positive')
        self.capacity_rate = self.volume_flow_L_h / 3600 * CP_J_KG_K
        d = self.reference_power_W / self.capacity_rate
        theta = self.reference_supply_C - self.reference_room_C
        if theta <= d:
            raise ValueError('Radiator reference return must exceed reference room temperature')
        if self.max_power_W < self.reference_power_W or self.power_step_W > self.max_power_W:
            raise ValueError('Radiator map range must include the reference power and at least one step')
        if self.max_power_W / self.power_step_W > 1000000:
            raise ValueError('Radiator map exceeds one million intervals')
        self.reference_log_mean_K = d / np.log1p(d / (theta - d))
        self.power_W = np.unique(np.append(np.linspace(0, self.max_power_W,
            int(np.ceil(self.max_power_W / self.power_step_W)) + 1), self.reference_power_W))
        positive = self.power_W[1:]
        spread = positive / self.capacity_rate
        mean = self.reference_log_mean_K * (positive / self.reference_power_W) ** (1 / self.exponent)
        ratio = spread / mean
        # Equivalent form avoids overflow of exp(ratio) for large ratios.
        self.return_K = np.r_[0., spread * np.exp(-ratio) / (-np.expm1(-ratio))]
        self.supply_K = self.return_K + self.power_W / self.capacity_rate

    def temperatures(self, power_W):
        if not np.isfinite(power_W) or power_W < 0 or power_W > self.max_power_W:
            raise ValueError('Radiator demand outside configured map power range')
        supply = float(np.interp(power_W, self.power_W, self.supply_K))
        return supply, supply - power_W / self.capacity_rate

    def flow(self, hot_C, room_C, demand_W):
        from models.hydraulics import HeatingFlow
        if not all(np.isfinite(value) for value in (hot_C, room_C, demand_W)):
            raise ValueError("Radiator temperatures and demand must be finite")
        requested = max(0., demand_W)
        target_supply, _ = self.temperatures(requested)
        available = max(0., hot_C - room_C)
        power = min(requested, float(np.interp(available, self.supply_K, self.power_W)))
        supply, ret = self.temperatures(power)
        fraction = (supply - ret) / (hot_C - room_C - ret) if power > 0 else 0.
        mass = self.volume_flow_L_h / 3600
        return HeatingFlow(room_C + supply, room_C + ret, power, mass,
                           mass * fraction, fraction, room_C + target_supply,
                           room_C + self.temperatures(requested)[1])
