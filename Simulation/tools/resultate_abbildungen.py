"""Alle Abbildungen und Tabellen für das Kapitel Resultate.

Start im Projektordner:
    python tools/resultate_abbildungen.py [Studienordner] [--out ../Doc/Resultate]

Ohne Studienordner wird der Hauptvergleich der Arbeit (results/hauptvergleich) verwendet. Die
Validierungsläufe werden aus results/validation gelesen; wurde eine Validierung neu gerechnet, wird
der neueste study_-Ordner darin verwendet, sonst die Ergebnisse der Arbeit. Ausgabe: PDF-Abbildungen und
LaTeX-Tabellen (\\input{...}) im Ausgabeordner, dazu zerlegung.csv mit allen Zahlen der Zerlegung.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Feste Reihenfolge, Farbe und Basis jeder Strategie (Referenz grau)
STRATEGIES = {
    'heating_curve': ('#7f7f7f', None),
    'heating_curve_supply': ('#1f6fb4', 'heating_curve'),
    'pause_load_shifting': ('#2ca02c', 'heating_curve_supply'),
    'run_load_shifting': ('#9467bd', 'heating_curve_supply'),
    'reinforcement_learning': ('#d62728', 'heating_curve'),
    'reinforcement_learning_offset': ('#e08a1e', 'heating_curve'),
}
SHORT = {
    'heating_curve': 'heating\\_curve', 'heating_curve_supply': 'heating\\_curve\\_supply',
    'pause_load_shifting': 'pause\\_load\\_shifting', 'run_load_shifting': 'run\\_load\\_shifting',
    'reinforcement_learning': 'reinforcement\\_learning',
    'reinforcement_learning_offset': 'reinforcement\\_learning\\_offset',
}
HEATING_MONTHS = (9, 10, 11, 12, 1, 2, 3, 4, 5)  # aktiver Heizzeitraum: September bis Ende Mai

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.3, 'legend.frameon': False})


def de(x, digits=0):
    """Zahl im Schweizer Format für LaTeX: Komma, schmales Leerzeichen als Tausendertrennung."""
    s = f'{x:,.{digits}f}'.replace(',', 'X').replace('.', ',').replace('X', '\\,')
    return s.replace('-', '$-$') if s.startswith('-') else s


def signed(x, digits=0):
    return ('+' if x > 0 else '') + de(x, digits)


def latest_study(folder: Path) -> Path:
    """Neuester study_-Ordner in folder (eigener neuer Lauf), sonst folder selbst (Ergebnisse der Arbeit)."""
    studies = sorted(folder.glob('study_*'))
    if not studies and (folder / 'summary.csv').is_file():
        return folder
    if not studies:
        raise FileNotFoundError(f'Kein study_-Ordner in {folder}')
    return studies[-1]


def read_summary(study: Path) -> list[dict]:
    with (study / 'summary.csv').open(encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def num(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return float('nan')


COLUMNS = ['timestamp', 'outside_temperature_C', 'room_temperature_C', 'target_room_temperature_C',
           'storage_control_temperature_C', 'thermal_power_W', 'heating_power_to_building_W',
           'electric_power_W', 'heat_pump_power_fraction', 'wp_supply_temperature_C', 'hk_shift_K']


def read_series(run_dir: Path) -> dict:
    data = {c: [] for c in COLUMNS}
    with (run_dir / 'simulation_results.csv').open(encoding='utf-8-sig', newline='') as fh:
        for row in csv.DictReader(fh):
            for c in COLUMNS:
                data[c].append(row.get(c, ''))
    out = {'time': [datetime.fromisoformat(t) for t in data.pop('timestamp')]}
    for c, values in data.items():
        out[c] = np.array([float(v) if v not in ('', None) else np.nan for v in values])
    out['hour'] = np.array([t.hour + t.minute / 60 for t in out['time']])
    out['month'] = np.array([t.month for t in out['time']])
    return out


def hourly_mean(ts, values, mask):
    """Mittelwert je Stunde des Tages (0..23) über die maskierten Zeitpunkte."""
    h = np.floor(ts['hour'][mask]).astype(int)
    v = values[mask]
    return np.array([np.nanmean(v[h == k]) for k in range(24)])


# ---------------------------------------------------------------- Abbildungen

def fig_tagesverlauf(series, out):
    """A1: mittlerer Tagesverlauf der Heizperiode für die beiden Grundregelungen."""
    fig, axes = plt.subplots(3, 1, figsize=(6.3, 6.4), sharex=True)
    x = np.arange(24) + 0.5
    for name in ('heating_curve', 'heating_curve_supply'):
        ts, color = series[name], STRATEGIES[name][0]
        m = np.isin(ts['month'], HEATING_MONTHS)
        axes[0].plot(x, hourly_mean(ts, ts['room_temperature_C'], m), color=color, label=name)
        axes[1].plot(x, hourly_mean(ts, ts['storage_control_temperature_C'], m), color=color, label=name)
        axes[2].plot(x, hourly_mean(ts, ts['thermal_power_W'], m) / 1000, color=color, label=f'WP, {name}')
    ts = series['heating_curve']
    m = np.isin(ts['month'], HEATING_MONTHS)
    axes[0].plot(x, hourly_mean(ts, ts['target_room_temperature_C'], m), 'k--', lw=1, label='Raum-Solltemperatur')
    axes[2].plot(x, hourly_mean(ts, ts['heating_power_to_building_W'], m) / 1000, 'k--', lw=1,
                 label='an das Gebäude abgegeben (heating_curve)')
    axes[0].set_ylabel('Raumtemperatur [°C]')
    axes[1].set_ylabel('Speicherfühler [°C]')
    axes[2].set_ylabel('Wärmeleistung [kW]')
    axes[2].set_xlabel('Tageszeit [h]')
    axes[2].set_xticks(range(0, 25, 3))
    for ax in axes:
        ax.axvspan(0, 6, color='0.9', zorder=0, lw=0)
        ax.axvspan(21, 24, color='0.9', zorder=0, lw=0)
        ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(out / 'A1_tagesverlauf.pdf')
    plt.close(fig)


def fig_score_anteile(rows, out):
    """A2: Score-Unterschied jeder Strategie gegenüber ihrer Basis, aufgeteilt in drei Teile."""
    by = {r['control.method']: r for r in rows}
    names = [n for n in STRATEGIES if STRATEGIES[n][1] and n in by]
    parts = [('electricity_cost', 'Strom', '#4c72b0'), ('cycle_cost', 'Starts', '#dd8452'),
             ('comfort_cost', 'Komfort', '#8c8c8c')]
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    y = np.arange(len(names))[::-1]
    for yi, n in zip(y, names):
        base = by[STRATEGIES[n][1]]
        neg = pos = 0.0
        for key, label, color in parts:
            d = num(by[n], key) - num(base, key)
            left = neg if d < 0 else pos
            ax.barh(yi, d, left=left, color=color, height=0.6, label=label if yi == y[0] else None)
            if d < 0:
                neg += d
            else:
                pos += d
        total = num(by[n], 'total_score') - num(base, 'total_score')
        ax.plot(total, yi, 'kD', ms=5, label='Total' if yi == y[0] else None)
        ax.annotate(f'{total:+.0f}', (min(neg, total), yi), xytext=(-4, 0), textcoords='offset points',
                    ha='right', va='center', fontsize=8)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_yticks(y, [f'{n}\n(Basis: {STRATEGIES[n][1]})' for n in names], fontsize=7)
    ax.set_xlabel('Unterschied zum Total Score der Basis [CHF/Jahr], negativ = besser')
    lo = ax.get_xlim()[0]
    ax.set_xlim(lo * 1.15, ax.get_xlim()[1])
    ax.legend(fontsize=7, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4)
    fig.tight_layout()
    fig.savefig(out / 'A2_score_anteile.pdf')
    plt.close(fig)


def attribution_all(study, run_of, out):
    """Zerlegung jeder Strategie gegenüber ihrer Basis; schreibt zerlegung.csv und gibt die Werte zurück."""
    from analysis.efficiency_attribution import compute, load_run
    results, weighted = {}, {}
    for name, run in run_of.items():
        A, _, _ = load_run(run)
        w = A['Q'] * A['on']
        weighted[name] = (float((w * A['To']).sum() / w.sum()), float(np.nansum(w * A['Ts']) / w[np.isfinite(A['Ts'])].sum()))
    for name, (_, base) in STRATEGIES.items():
        if base and name in run_of and base in run_of:
            results[name] = compute(run_of[base], run_of[name])
    with (out / 'zerlegung.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['strategie', 'basis', 'differenz_strom_CHF'] + [t['key'] for t in next(iter(results.values()))['terms']]
                   + ['T_aussen_waermegewichtet_C', 'T_vorlauf_waermegewichtet_C'])
        for name in STRATEGIES:
            if name in results:
                r = results[name]
                w.writerow([name, STRATEGIES[name][1], round(r['difference_cost'], 2)]
                           + [round(t['value'], 2) for t in r['terms']] + [round(v, 3) for v in weighted[name]])
            elif name in weighted:
                w.writerow([name, '', '', *[''] * 7, *[round(v, 3) for v in weighted[name]]])
    return results, weighted


def fig_zerlegung(results, out):
    """A3: Stromkostenunterschied gegenüber der Basis, aufgeteilt auf die Ursachen."""
    keys = [('timing', 'Timing'), ('temperature', 'Temperatur-\nniveau'),
            ('power_steps', 'Leistungs-\nstufen')]
    keys.extend([('quantity', 'Wärme-\nmenge'), ('residual', 'Rest')])
    names = [n for n in STRATEGIES if n in results]
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    width = 0.8 / len(names)
    x = np.arange(len(keys) + 1)
    for i, n in enumerate(names):
        terms = {t['key']: t['value'] for t in results[n]['terms']}
        # Very small differences from days with little or no heat production
        # are included in the residual so that the displayed terms still sum.
        terms['residual'] = terms.get('residual', 0.0) + terms.get('low_heat_days', 0.0)
        vals = [terms.get(k, 0.0) for k, _ in keys]
        vals.append(results[n]['difference_cost'])
        ax.bar(x + (i - (len(names) - 1) / 2) * width, vals, width, color=STRATEGIES[n][0], label=n)
    ax.set_axisbelow(True)
    ax.grid(axis='y')
    ax.grid(axis='x', visible=False)
    ax.axhline(0, color='k', lw=0.8)
    # Category separators belong in the gaps between bars, not on the tick
    # positions. The divider before the total is kept slightly more visible.
    for boundary in np.arange(0.5, len(keys) - 0.5, 1.0):
        ax.axvline(boundary, color='0.82', lw=0.6, zorder=0)
    ax.axvline(len(keys) - 0.5, color='0.5', lw=0.8, ls=':', zorder=0)
    ax.set_xticks(x, [l for _, l in keys] + ['Summe'], fontsize=8)
    ax.set_ylabel('Stromkosten gegenüber Basis [CHF/Jahr]')
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / 'A3_zerlegung.pdf')
    plt.close(fig)


def fig_timing_temperatur(weighted, out):
    """A4: wärmegewichtete Aussentemperatur gegen wärmegewichtete WP-Vorlauftemperatur."""
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for name, (color, base) in STRATEGIES.items():
        if name not in weighted:
            continue
        x, y = weighted[name]
        ax.plot(x, y, 'o', color=color, ms=7, label=name)
        if base in weighted:
            bx, by_ = weighted[base]
            ax.annotate('', (x, y), (bx, by_), arrowprops=dict(arrowstyle='->', color=color, lw=1))
    ax.set_xlabel('Wärmegewichtete Aussentemperatur [°C]\n(weiter rechts = Wärme bei wärmerer Luft erzeugt)')
    ax.set_ylabel('Wärmegewichtete WP-Vorlauftemperatur [°C]')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'A4_timing_temperatur.pdf')
    plt.close(fig)


def fig_waerme_tageszeit(series, out):
    """A5: Anteil der erzeugten Wärme je Tagesstunde in der Heizperiode."""
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    x = np.arange(24) + 0.5
    for name, ts in series.items():
        m = np.isin(ts['month'], HEATING_MONTHS)
        h = np.floor(ts['hour'][m]).astype(int)
        q = np.nan_to_num(ts['thermal_power_W'][m])
        share = np.array([q[h == k].sum() for k in range(24)]) / q.sum() * 100
        ax.plot(x, share, color=STRATEGIES[name][0], lw=2 if name == 'heating_curve' else 1.3, label=name)
    ax.axvspan(0, 6, color='0.9', zorder=0, lw=0)
    ax.axvspan(21, 24, color='0.9', zorder=0, lw=0)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel('Tageszeit [h] (grau: Nachtabsenkung)')
    ax.set_ylabel('Anteil der Wärme [%]')
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / 'A5_waerme_tageszeit.pdf')
    plt.close(fig)


def fig_leistungsstufen(series, out):
    """A6: Anteil der erzeugten Wärme je Leistungsstufe."""
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    levels = np.round(np.arange(0, 1.0001, 0.05), 2)
    for name, ts in series.items():
        f = np.round(ts['heat_pump_power_fraction'], 2)
        q = np.nan_to_num(ts['thermal_power_W'])
        on = q > 0
        share = np.array([q[on & np.isclose(f, l)].sum() for l in levels]) / q[on].sum() * 100
        ax.plot(levels * 100, share, marker='o', ms=3, color=STRATEGIES[name][0],
                lw=2 if name == 'heating_curve' else 1.2, label=name)
    ax.set_xlabel('Leistungsstufe [%] (0 = kleinste Verdichterleistung)')
    ax.set_ylabel('Anteil der Wärme [%]')
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / 'A6_leistungsstufen.pdf')
    plt.close(fig)


def fig_rl(series, out):
    """A7: Was die beiden RL-Regelungen im Tagesverlauf tun."""
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 4.4), sharex=True)
    x = np.arange(24) + 0.5
    for name in ('heating_curve', 'reinforcement_learning'):
        if name in series:
            ts = series[name]
            m = np.isin(ts['month'], HEATING_MONTHS)
            on_share = hourly_mean(ts, (np.nan_to_num(ts['thermal_power_W']) > 0).astype(float), m) * 100
            axes[0].plot(x, on_share, color=STRATEGIES[name][0], label=f'{name}: Laufzeitanteil')
            frac = np.where(ts['thermal_power_W'] > 0, ts['heat_pump_power_fraction'], np.nan) * 100
            axes[0].plot(x, hourly_mean(ts, frac, m), color=STRATEGIES[name][0], ls='--',
                         label=f'{name}: mittlere Stufe im Betrieb')
    if 'reinforcement_learning_offset' in series:
        ts = series['reinforcement_learning_offset']
        m = np.isin(ts['month'], HEATING_MONTHS)
        axes[1].plot(x, hourly_mean(ts, ts['hk_shift_K'], m), color=STRATEGIES['reinforcement_learning_offset'][0],
                     label='reinforcement_learning_offset')
        axes[1].axhline(0, color='k', lw=0.8)
    axes[0].set_ylabel('[%]')
    axes[1].set_ylabel('Mittlere Verschiebung\nWP HK [K]')
    axes[1].set_xlabel('Tageszeit [h] (grau: Nachtabsenkung)')
    axes[1].set_xticks(range(0, 25, 3))
    for ax in axes:
        ax.axvspan(0, 6, color='0.9', zorder=0, lw=0)
        ax.axvspan(21, 24, color='0.9', zorder=0, lw=0)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'A7_rl_tagesverlauf.pdf')
    plt.close(fig)


# ---------------------------------------------------------------- Tabellen

def write(out, name, lines):
    (out / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def tab_haupt(rows, out):
    by = {r['control.method']: r for r in rows}
    lines = [r'\begin{tabular}{l r r r r r r r}', r'\toprule',
             r'Strategie & Strom & Wärme & AZ & Starts & Komfort & Score & $\Delta$Score\\',
             r' & [kWh] & [kWh] & [--] & [1/d] & [K\,h] & [CHF] & [CHF]\\', r'\midrule']
    for name, (_, base) in STRATEGIES.items():
        if name not in by:
            continue
        r = by[name]
        E, Q = num(r, 'total_electric_energy_Wh') / 1000, num(r, 'total_thermal_energy_Wh') / 1000
        delta = signed(num(r, 'total_score') - num(by[base], 'total_score')) if base else '--'
        lines.append(rf'\texttt{{{SHORT[name]}}} & {de(E)} & {de(Q)} & {de(Q / E, 2)} & '
                     rf'{de(num(r, "heat_pump_starts_per_day"), 1)} & {de(num(r, "comfort_degree_hours"), 1)} & '
                     rf'{de(num(r, "total_score"))} & {delta}\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write(out, 'T2_hauptvergleich.tex', lines)


def tab_versuche(rows, out):
    by = {r['control.method']: r for r in rows}
    lines = [r'\begin{tabular}{l r r r r r}', r'\toprule',
             r'Strategie & Versuche & angenommen & keine Ersparnis & keine Rückkehr & nicht machbar\\', r'\midrule']
    for name in ('pause_load_shifting', 'run_load_shifting'):
        if name in by:
            r = by[name]
            lines.append(rf'\texttt{{{SHORT[name]}}} & ' + ' & '.join(
                de(num(r, k)) for k in ('trials_total', 'trials_accepted', 'trials_no_saving',
                                        'trials_no_recovery', 'trials_infeasible')) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write(out, 'T3_versuche.tex', lines)


def tab_zeitschritt(validation, out):
    folder = validation / '01_zeitschritt'
    if not folder.is_dir():
        return
    rows = [r for r in read_summary(latest_study(folder)) if r['kind'] == 'variant']
    lines = [r'\begin{tabular}{r r r r r r}', r'\toprule',
             r'Rechenschritt [min] & Strom [kWh] & Starts & Score [CHF] & Abw. Strom & Abw. Score\\',
             r'\midrule']
    for name in ('heating_curve',):  # nur die Referenz: geprüft wird die physikalische Rechnung
        sub = sorted((r for r in rows if r['control.method'] == name), key=lambda r: float(r['simulation.dt_minutes']))
        if not sub:
            continue
        ref = sub[0]
        for r in sub:
            dE = (num(r, 'total_electric_energy_Wh') / num(ref, 'total_electric_energy_Wh') - 1) * 100
            dS = (num(r, 'total_score') / num(ref, 'total_score') - 1) * 100
            lines.append(rf'{r["simulation.dt_minutes"]} & '
                         rf'{de(num(r, "total_electric_energy_Wh") / 1000)} & {de(num(r, "heat_pump_start_count"))} & '
                         rf'{de(num(r, "total_score"))} & {signed(dE, 2)}\,\% & {signed(dS, 2)}\,\%\\')
        lines.append(r'\addlinespace')
    lines = lines[:-1] + [r'\bottomrule', r'\end{tabular}']
    write(out, 'T1_zeitschritt.tex', lines)


def tab_sensitivitaet(validation, out):
    folder = validation / '03_verlustanteil'
    if folder.is_dir():
        rows = [r for r in read_summary(latest_study(folder)) if r['kind'] == 'variant']
        fs = sorted({float(r['building.direct_loss_fraction']) for r in rows})
        lines = [r'\begin{tabular}{l ' + 'r ' * len(fs) + '}', r'\toprule',
                 r'Strategie & ' + ' & '.join(rf'$f={de(f, 1)}$' for f in fs) + r'\\', r'\midrule']
        for name in STRATEGIES:
            cells = []
            for f in fs:
                match = [r for r in rows if r['control.method'] == name and float(r['building.direct_loss_fraction']) == f]
                cells.append(f'{de(num(match[0], "total_score"))} ({de(num(match[0], "total_electric_energy_Wh") / 1000)})'
                             if match else '--')
            if any(c != '--' for c in cells):
                lines.append(rf'\texttt{{{SHORT[name]}}} & ' + ' & '.join(cells) + r'\\')
        lines += [r'\bottomrule', r'\end{tabular}']
        write(out, 'T4a_verlustanteil.tex', lines)
    lines = [r'\begin{tabular}{l r r r r r}', r'\toprule',
             r'Strategie & Toleranz [K] & Score [CHF] & Komfort [K\,h] & Versuche & angenommen\\', r'\midrule']
    for sub, name in (('pause', 'pause_load_shifting'), ('run', 'run_load_shifting')):
        folder = validation / '04_toleranzen' / sub
        if not folder.is_dir():
            continue
        key = f'control.{name}.room_tolerance_K'
        rows = sorted((r for r in read_summary(latest_study(folder)) if r['kind'] == 'variant'), key=lambda r: float(r[key]))
        for r in rows:
            lines.append(rf'\texttt{{{SHORT[name]}}} & {de(float(r[key]), 2)} & {de(num(r, "total_score"))} & '
                         rf'{de(num(r, "comfort_degree_hours"), 1)} & {de(num(r, "trials_total"))} & {de(num(r, "trials_accepted"))}\\')
        lines.append(r'\addlinespace')
    lines = lines[:-1] + [r'\bottomrule', r'\end{tabular}']
    write(out, 'T4b_toleranzen.tex', lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('study', nargs='?', help='Studienordner des Hauptvergleichs')
    ap.add_argument('--out', default=str(ROOT.parent / 'Doc' / 'Resultate'))
    ap.add_argument('--validation', default=str(ROOT / 'results' / 'validation'))
    a = ap.parse_args()
    study = Path(a.study) if a.study else ROOT / 'results' / 'hauptvergleich'
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_summary(study) if r['status'] == 'completed']
    seen, run_of = set(), {}
    for r in rows:  # Basislauf und gleichnamige Variante sind identisch; die erste Zeile genügt
        m = r['control.method'] if r.get('control.method') else 'heating_curve'
        r['control.method'] = m
        if m not in seen:
            seen.add(m)
            run_of[m] = study / r['run_id']
    rows = [r for r in rows if study / r['run_id'] in run_of.values()]
    print('Studie:', study)
    series = {n: read_series(run_of[n]) for n in STRATEGIES if n in run_of}
    print('Zeitreihen gelesen')
    fig_tagesverlauf(series, out)
    fig_score_anteile(rows, out)
    results, weighted = attribution_all(study, run_of, out)
    fig_zerlegung(results, out)
    fig_timing_temperatur(weighted, out)
    fig_waerme_tageszeit(series, out)
    fig_leistungsstufen(series, out)
    fig_rl(series, out)
    tab_haupt(rows, out)
    tab_versuche(rows, out)
    validation = Path(a.validation)
    tab_zeitschritt(validation, out)
    tab_sensitivitaet(validation, out)
    print('Ausgabe in', out)


if __name__ == '__main__':
    main()
