"""Attribute the electricity difference between two saved runs to physical causes.

Usage (from the project folder):
    python -m analysis.efficiency_attribution <baseline_run_dir> <compare_run_dir> [--from DATE] [--to DATE]

A run dir is a folder with ``simulation_results.csv`` and ``config.yaml`` (e.g.
results/experiments/study_.../run_001). The electricity-cost difference is split per day
(Shapley over three operating factors of the specific electricity use e, in kWh per kWh of heat):
  * Timing        : heat-weighted outside temperature (real load shifting)
  * Temperature   : heat-weighted heat-pump supply temperature (storage level)
  * Power steps   : distribution of the power fractions (efficient part-load levels)
plus quantity (more/less heat produced), residual (unattributed difference) and
days with less than 2 kWh of heat or without electricity consumption. The values are electricity costs of the scoring (without start and comfort
costs) in the scoring currency. Model-based and not unique; the COP map of the compare run is used.
"""
from __future__ import annotations

import argparse
import csv
import itertools
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

import numpy as np

from simulation.config import load_config
from analysis.scoring import ScoringParams
from models.heat_pump import get_performance
from simulation.engine import build_simulation_params


def _parse_time(value):
    return datetime.fromisoformat(str(value).replace('T', ' ')) if value else None


def load_run(run_dir, start=None, end=None):
    """Read simulation_results.csv + config.yaml; ``start``/``end`` (inclusive) limit the window."""
    run_dir = Path(run_dir)
    start, end = _parse_time(start), _parse_time(end)
    rows = []
    with (run_dir / 'simulation_results.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        node_cols = sorted((c for c in reader.fieldnames if c.startswith('storage_node_')),
                           key=lambda c: int(c.split('_')[2]))
        for row in reader:
            t = datetime.fromisoformat(row['timestamp'])
            if (start is None or t >= start) and (end is None or t <= end):
                rows.append((t, row))
    if len(rows) < 2:
        raise ValueError('Zu wenige Zeitschritte im gewählten Zeitraum')

    def col(name):
        out = []
        for _, r in rows:
            try:
                out.append(float(r[name]))
            except (ValueError, TypeError, KeyError):
                out.append(np.nan)
        return np.array(out)

    times = [t for t, _ in rows]
    intervals_s = np.array([(b - a).total_seconds() for a, b in zip(times, times[1:])], dtype=float)
    dt_s = float(np.median(intervals_s))
    if not np.allclose(intervals_s, dt_s, rtol=0.0, atol=1e-6):
        raise ValueError('The attribution requires regularly spaced simulation output')
    dt_h = dt_s / 3600
    A = dict(
        day=np.array([t.strftime('%Y-%m-%d') for t in times]),
        timestamps=np.array([t.isoformat() for t in times]),
        To=col('outside_temperature_C'), P=np.nan_to_num(col('electric_power_W')),
        Q=np.nan_to_num(col('thermal_power_W')), Ts=col('wp_supply_temperature_C'),
        f=np.nan_to_num(col('heat_pump_power_fraction')), Qb=np.nan_to_num(col('heating_power_to_building_W')),
        loss=np.nan_to_num(col('storage_loss_W')), Ustor=col('storage_energy_Wh'),
        Ttop=col(node_cols[0]) if node_cols else None, Tbot=col(node_cols[-1]) if node_cols else None,
    )
    A['on'] = A['P'] > 0
    cfg = load_config(str(run_dir / 'config.yaml'))
    scoring = ScoringParams.from_config(cfg['scoring'])
    duration = timedelta(hours=dt_h)
    A['C'] = np.array([scoring.step_cost(t.isoformat(), duration, max(0.0, p))
                       for t, p in zip(times, A['P'])])
    A['currency'] = scoring.currency
    A['starts'] = int((A['on'][1:] & ~A['on'][:-1]).sum()) + int(A['on'][0])
    return A, dt_h, cfg


def balance(A, dt_h):
    k = dt_h / 1000
    d = dict(E_el=A['P'].sum() * k, cost=A['C'].sum(), Q_hp=A['Q'].sum() * k, Q_bldg=A['Qb'].sum() * k, loss=A['loss'].sum() * k,
             dU_stor=(A['Ustor'][-1] - A['Ustor'][0]) / 1000)
    d['COP'] = d['Q_hp'] / d['E_el']
    return d


def attribute(Ab, Ar, dt_h, hp):
    def cop(To, Ts, f):
        return np.array([get_performance(a, b, hp, max(0.0, c))[0]
                         for a, b, c in zip(np.atleast_1d(To), np.atleast_1d(Ts), np.atleast_1d(f))])
    k = dt_h / 1000
    tot = defaultdict(float)
    names = ['timing', 'temperature', 'power_steps']
    for d in sorted(set(Ab['day']) & set(Ar['day'])):
        mb = (Ab['day'] == d) & Ab['on']
        mr = (Ar['day'] == d) & Ar['on']
        Qb, Qr = Ab['Q'][mb].sum() * k, Ar['Q'][mr].sum() * k
        Eb, Er = Ab['P'][mb].sum() * k, Ar['P'][mr].sum() * k
        Cb, Cr = Ab['C'][mb].sum(), Ar['C'][mr].sum()
        if Qb < 2 or Qr < 2 or Eb <= 0 or Er <= 0:
            tot['low_heat_days'] += Cr - Cb
            continue
        price_b, price_r = Cb / Eb, Cr / Er
        if not np.isclose(price_b, price_r, rtol=1e-9, atol=1e-12):
            raise ValueError('The three-factor attribution requires the same constant electricity price in both runs')
        cb, cr = Cb / Qb, Cr / Qr  # electricity cost per kWh of heat
        src = {}
        for tag, A, m, E, C in (('b', Ab, mb, Eb, Cb), ('r', Ar, mr, Er, Cr)):
            w = A['Q'][m] / A['Q'][m].sum()
            src[tag] = dict(To=(w * A['To'][m]).sum(), Ts=(w * A['Ts'][m]).sum(), w=w, f=A['f'][m])
        e = {}
        for c in itertools.product('br', repeat=3):
            F = src[c[2]]
            e[c] = (F['w'] / cop(src[c[0]]['To'], src[c[1]]['Ts'], F['f'])).sum()
        # Relative model change applied to the measured baseline value: a COP map that deviates from
        # the simulated COP in absolute terms does not leak into the factors.
        eb_measured = Eb / Qb
        v = {c: eb_measured * e[c] / e[('b', 'b', 'b')]
             for c in itertools.product('br', repeat=3)}
        contrib = dict.fromkeys(names, 0.0)
        orders = list(itertools.permutations(range(3)))
        for perm in orders:
            cur = ['b'] * 3
            for i in perm:
                before = v[tuple(cur)]
                cur[i] = 'r'
                contrib[names[i]] += (v[tuple(cur)] - before) / len(orders)
        tot['quantity'] += cb * (Qr - Qb)
        for n in names:
            tot[n] += Qr * price_b * contrib[n]
        tot['residual'] += Qr * ((cr - cb) - price_b * sum(contrib.values()))
    return dict(tot)


LABELS = [('temperature', 'Temperaturniveau (Speicher/Vorlauf)'), ('power_steps', 'Leistungsstufen'),
          ('timing', 'Timing (Aussentemperatur, Lastverschiebung)'),
          ('quantity', 'Wärmemenge'), ('residual', 'Wechselwirkung/Rest'),
          ('low_heat_days', 'Tage mit geringer Wärmeerzeugung')]


def compute(baseline_dir, compare_dir, start=None, end=None):
    """Return a JSON-friendly attribution of the electricity-cost difference compare - baseline."""
    Ab, dtb, _ = load_run(baseline_dir, start, end)
    Ar, dtr, cfg = load_run(compare_dir, start, end)
    if not np.isclose(dtb, dtr) or not np.array_equal(Ab['timestamps'], Ar['timestamps']):
        raise ValueError('Runs need identical timestamps and time steps for a day-by-day attribution')
    hp = build_simulation_params(cfg).heat_pump
    bb, br = balance(Ab, dtb), balance(Ar, dtr)
    terms = attribute(Ab, Ar, dtr, hp)
    on_b, on_r = Ab['on'] & np.isfinite(Ab['Ts']), Ar['on'] & np.isfinite(Ar['Ts'])
    return dict(
        electricity_kWh=[bb['E_el'], br['E_el']], heat_kWh=[bb['Q_hp'], br['Q_hp']],
        cop=[bb['COP'], br['COP']], starts=[Ab['starts'], Ar['starts']],
        storage_loss_kWh=[bb['loss'], br['loss']], storage_energy_change_kWh=[bb['dU_stor'], br['dU_stor']],
        supply_temperature_C=[float(Ab['Ts'][on_b].mean()) if on_b.any() else None,
                              float(Ar['Ts'][on_r].mean()) if on_r.any() else None],
        difference_kWh=br['E_el'] - bb['E_el'],
        currency=Ar['currency'], cost=[bb['cost'], br['cost']], difference_cost=br['cost'] - bb['cost'],
        mean_price_per_kWh=[bb['cost'] / bb['E_el'], br['cost'] / br['E_el']],
        terms=[dict(key=k, label=n, value=float(terms.get(k, 0.0))) for k, n in LABELS],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('baseline'); ap.add_argument('compare')
    ap.add_argument('--from', dest='start'); ap.add_argument('--to', dest='end')
    a = ap.parse_args()
    r = compute(a.baseline, a.compare, a.start, a.end)
    dC, cur = r['difference_cost'], r['currency']
    e, q, c, s_ = r['electricity_kWh'], r['heat_kWh'], r['cop'], r['starts']
    cost, price = r['cost'], r['mean_price_per_kWh']
    print(f"Stromkosten Basis {cost[0]:.2f} {cur} -> Vergleich {cost[1]:.2f} {cur}  ({dC:+.2f} {cur}, {100*dC/cost[0]:+.1f} %)")
    print(f"Strom {e[0]:.1f} -> {e[1]:.1f} kWh | mittlerer Preis {price[0]:.3f} -> {price[1]:.3f} {cur}/kWh")
    print(f"Waerme Basis {q[0]:.1f} -> {q[1]:.1f} kWh | COP {c[0]:.2f} -> {c[1]:.2f} | Starts {s_[0]} -> {s_[1]}")
    print(f"WP-Vorlauf (Betrieb) {r['supply_temperature_C'][0]:.1f} -> {r['supply_temperature_C'][1]:.1f} C")
    print(f"{'Anteil':45s} {cur:>8s} {'% der Differenz':>16s}")
    for t in r['terms']:
        print(f"{t['label']:45s} {t['value']:8.2f} {100*t['value']/dC if dC else float('nan'):15.0f} %")


if __name__ == '__main__':
    main()
