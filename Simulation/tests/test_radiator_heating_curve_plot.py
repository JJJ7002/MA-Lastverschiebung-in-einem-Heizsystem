from pathlib import Path
import sys

import pytest
from models.building import BuildingParams

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support import load_test_config
from analysis.visualize import calculate_radiator_heating_curve, try_plot_heating_curve
from models.radiator import RadiatorMap
from models.stratified_tank import CP_J_KG_K


def configuration():
    return load_test_config()


def test_curve_preserves_stationary_load_and_water_balance():
    config = configuration()
    radiator = RadiatorMap(volume_flow_L_h=config['heating']['volume_flow_L_h'],
                           **config['heating']['radiator'])
    capacity = config['heating']['volume_flow_L_h'] / 3600 * CP_J_KG_K
    rows = calculate_radiator_heating_curve(config)
    for row in rows:
        room, outside = row['room_target_C'], row['outside_temp_C']
        heat = max(0, BuildingParams.from_config(config['building']).stationary_UA_W_per_K * (room-outside))
        assert row['heating_demand_W'] == pytest.approx(heat)
        supply, ret = radiator.temperatures(heat)
        assert row['heating_supply_target_C'] == pytest.approx(room+supply)
        assert row['heating_return_target_C'] == pytest.approx(room+ret)
        assert capacity * (row['heating_supply_target_C']-row['heating_return_target_C']) == pytest.approx(heat)
    assert all(a['heating_supply_target_C'] >= b['heating_supply_target_C']
               for a, b in zip(rows, rows[1:]))


def test_curve_independent_of_legacy_curves_and_initial_mass():
    config = configuration()
    config['control']['heating_curve']['slope_factor'] = 1.0
    expected = calculate_radiator_heating_curve(config)
    offset = config['control'].pop('heating_curve')['offset_K']
    config['control']['heating_curve'] = {'slope_factor': 1.0, 'offset_K': offset}
    config['control']['heating_circuit_curve'] = {'slope': 999, 'offset_C': -999}
    config['building']['T_mass_init_C'] = 5
    config['heating']['room_response_minutes'] = 60
    assert calculate_radiator_heating_curve(config) == expected


def test_plot_exports_curve_and_csv(tmp_path):
    output = tmp_path / 'heating_curve.png'
    assert try_plot_heating_curve(configuration(), output) == output
    assert output.stat().st_size > 0
    csv = (tmp_path / 'csv' / 'heating_curve.csv').read_text(encoding='utf-8')
    assert 'heating_demand_W' in csv
    assert 'T_on' not in csv
