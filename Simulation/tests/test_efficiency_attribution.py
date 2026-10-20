from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import load_test_config
from analysis.efficiency_attribution import attribute
from simulation.engine import build_simulation_params


def run(price, power_W=3000.0, heat_W=9000.0, steps=144, temp=2.0, supply=40.0, f=0.5):
    on = np.ones(steps, dtype=bool)
    P = np.full(steps, power_W)
    return dict(day=np.array(['2025-01-01'] * steps), To=np.full(steps, temp), P=P,
                Q=np.full(steps, heat_W), Ts=np.full(steps, supply), f=np.full(steps, f), on=on,
                C=P / 1000 * (10 / 60) * price)


def test_constant_price_is_not_a_separate_attribution_factor():
    hp = build_simulation_params(load_test_config()).heat_pump
    terms = attribute(run(0.30), run(0.30), 10 / 60, hp)
    cost_b = 144 * 3.0 * (10 / 60) * 0.30
    assert 'price' not in terms
    assert sum(terms.values()) == pytest.approx(0.0, abs=1e-9)


def test_less_electricity_at_same_price_is_attributed_to_operation_or_residual():
    hp = build_simulation_params(load_test_config()).heat_pump
    terms = attribute(run(0.30), run(0.30, power_W=2700.0), 10 / 60, hp)
    assert 'price' not in terms
    assert sum(terms.values()) == pytest.approx(144 * 0.3 * (10 / 60) * (2.7 - 3.0), rel=1e-6)


def test_attribution_closes_with_changes_in_operating_factors():
    hp = build_simulation_params(load_test_config()).heat_pump
    baseline = run(0.30)
    compare = run(0.30, power_W=2800.0)
    compare['To'] = np.full(144, 5.0)
    compare['Ts'] = np.full(144, 45.0)
    compare['f'] = np.full(144, 0.7)
    terms = attribute(baseline, compare, 10 / 60, hp)
    actual_cost_difference = compare['C'].sum() - baseline['C'].sum()
    assert sum(terms.values()) == pytest.approx(actual_cost_difference, abs=1e-9)


def test_days_below_heat_threshold_are_reported_separately():
    hp = build_simulation_params(load_test_config()).heat_pump
    baseline = run(0.30, heat_W=50.0)
    compare = run(0.30, power_W=2800.0, heat_W=50.0)
    terms = attribute(baseline, compare, 10 / 60, hp)
    actual_cost_difference = compare['C'].sum() - baseline['C'].sum()
    assert terms['low_heat_days'] == pytest.approx(actual_cost_difference)
    assert sum(terms.values()) == pytest.approx(actual_cost_difference)
