"""Optional numba kernels for the stratified storage (identical arithmetic to stratified_tank).

If numba is missing, or HEATSIM_NO_NUMBA=1 is set, ``KERNELS_AVAILABLE`` is False
and the numpy implementation in stratified_tank.py is used instead.
"""
import os

import numpy as np

try:
    if os.environ.get('HEATSIM_NO_NUMBA'):
        raise ImportError('numba disabled by HEATSIM_NO_NUMBA')
    from numba import njit
    KERNELS_AVAILABLE = True
except Exception:  # ImportError, or a numba/numpy version mismatch
    KERNELS_AVAILABLE = False

if KERNELS_AVAILABLE:
    @njit(cache=True)
    def mix_inversions(temps, caps):
        """Pool adjacent unstable blocks; preserve energy (stack of blocks)."""
        n = temps.shape[0]
        start = np.empty(n, np.int64)
        stop = np.empty(n, np.int64)
        cap = np.empty(n)
        energy = np.empty(n)
        top = -1
        for i in range(n):
            top += 1
            start[top] = i
            stop[top] = i + 1
            cap[top] = caps[i]
            energy[top] = caps[i] * temps[i]
            while top > 0 and energy[top - 1] / cap[top - 1] < energy[top] / cap[top]:
                stop[top - 1] = stop[top]
                cap[top - 1] = cap[top - 1] + cap[top]
                energy[top - 1] = energy[top - 1] + energy[top]
                top -= 1
        result = np.empty(n)
        for b in range(top + 1):
            value = energy[b] / cap[b]
            for i in range(start[b], stop[b]):
                result[i] = value
        return result

    @njit(cache=True)
    def tank_step(temps, caps, ua, cond, nodes, flows, inlets, cp, ambient, max_seconds, courant):
        """One CFL-limited explicit step. Returns (new, dt, port_heat, loss, change, residual, error)."""
        n = temps.shape[0]
        net = np.zeros(n)
        heat = np.zeros(n)
        outflow = np.zeros(n)
        for i in range(n):
            if not np.isfinite(temps[i]):
                return temps, 0.0, 0.0, 0.0, 0.0, 0.0, 1
        for p in range(nodes.shape[0]):
            node = nodes[p]
            if node < 0 or node >= n:
                return temps, 0.0, 0.0, 0.0, 0.0, 0.0, 2
            flow = flows[p]
            net[node] += flow
            if flow > 0:
                if np.isnan(inlets[p]):
                    return temps, 0.0, 0.0, 0.0, 0.0, 0.0, 4
                heat[node] += flow * cp * inlets[p]
            elif flow < 0:
                heat[node] += flow * cp * temps[node]
                outflow[node] -= flow
        total = 0.0
        for i in range(n):
            total += net[i]
        if abs(total) > 1e-9:
            return temps, 0.0, 0.0, 0.0, 0.0, 0.0, 3
        port_heat = 0.0
        for i in range(n):
            port_heat += heat[i]
        running = 0.0
        for i in range(n - 1):
            running += net[i]
            donor = i if running >= 0 else i + 1
            enthalpy = running * cp * temps[donor]
            heat[i] -= enthalpy
            heat[i + 1] += enthalpy
            outflow[donor] += abs(running)
        conduction = np.empty(max(n - 1, 0))
        for i in range(n - 1):
            conduction[i] = cond[i] * (temps[i] - temps[i + 1])
        for i in range(n - 1):
            heat[i] -= conduction[i]
        for i in range(n - 1):
            heat[i + 1] += conduction[i]
        loss = 0.0
        for i in range(n):
            node_loss = ua[i] * (temps[i] - ambient)
            heat[i] -= node_loss
            loss += node_loss
        limit = np.inf
        for i in range(n):
            diagonal = outflow[i] * cp + ua[i]
            if i < n - 1:
                diagonal += cond[i]
            if i > 0:
                diagonal += cond[i - 1]
            if diagonal > 0:
                ratio = caps[i] / diagonal
                if ratio < limit:
                    limit = ratio
        limit = courant * limit
        dt = max_seconds if max_seconds < limit else limit
        raw = np.empty(n)
        for i in range(n):
            raw[i] = temps[i] + heat[i] * dt / caps[i]
        new = mix_inversions(raw, caps)
        change = 0.0
        for i in range(n):
            change += caps[i] * (new[i] - temps[i])
        change = change / 3600
        residual = change - (port_heat - loss) * dt / 3600
        return new, dt, port_heat, loss, change, residual, 0


def warm_up():
    """Compile (or load from the on-disk cache) all kernels once."""
    if not KERNELS_AVAILABLE:
        return
    temps = np.linspace(60.0, 30.0, 10)
    caps = np.full(10, 1.0e5)
    tank_step(temps, caps, np.full(10, 0.1), np.full(9, 0.5), np.array([1, 9, 0, 8], np.int64),
              np.array([1.0, -1.0, -0.3, 0.3]), np.array([65.0, np.nan, np.nan, 32.0]),
              4182.0, 21.0, 10.0, 0.5)
