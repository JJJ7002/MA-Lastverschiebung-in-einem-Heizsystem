from __future__ import annotations

from pathlib import Path
from datetime import datetime
from math import ceil, isfinite
import csv
import os

from simulation.engine import SimulationResult
from analysis.output_paths import plot_csv_path


def plot_storage_stratification(result: SimulationResult, dt_hours: float, output_path: Path):
    """Show the layer history and the temperatures actually used by the plant."""
    if not result.storage_node_temperatures_C:
        return
    import numpy as np
    import matplotlib.pyplot as plt
    nodes = np.asarray(result.storage_node_temperatures_C)
    hours = (np.arange(len(nodes)) + 1) * dt_hours
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout='constrained')
    image = axes[0].imshow(nodes.T, aspect='auto', interpolation='nearest',
                           extent=(0, len(nodes) * dt_hours, nodes.shape[1] + .5, .5), cmap='inferno')
    axes[0].set_ylabel('Schicht (1 = oben)')
    fig.colorbar(image, ax=axes[0], label='Wassertemperatur [°C]')
    axes[1].plot(hours, nodes[:, 0], label='Puffer oben')
    axes[1].plot(hours, nodes[:, -1], label='Puffer unten')
    sensor_node = (int(result.hydraulics[0].get('storage_control_sensor_layer', max(1, nodes.shape[1] - 1))) - 1
                   if result.hydraulics else max(0, nodes.shape[1] - 2))
    axes[1].plot(hours, nodes[:, sensor_node], label=f'WP-Regelfühler (Schicht {sensor_node + 1})',
                 linestyle='-.')
    axes[1].plot(hours, nodes.mean(axis=1), label='Puffermittel', linestyle='--')
    if result.hydraulics:
        axes[1].plot(hours, [r['buffer_target_C'] for r in result.hydraulics], label='Puffersoll', alpha=.8)
        if 'wp_supply_target_temperature_C' in result.hydraulics[0]:
            axes[1].plot(hours, [r['wp_supply_target_temperature_C'] for r in result.hydraulics],
                         label='WP-Vorlaufsoll (Rücklauf + Hub)', alpha=.8, linestyle=':')
        if 'wp_supply_temperature_C' in result.hydraulics[0]:
            axes[1].plot(hours, [r['wp_supply_temperature_C'] for r in result.hydraulics],
                         label='WP-Vorlauf tatsächlich', alpha=.8)
        axes[1].plot(hours, [r['heating_supply_temperature_C'] for r in result.hydraulics], label='HK-Vorlauf', alpha=.8)
        if 'heating_supply_target_C' in result.hydraulics[0]:
            axes[1].plot(hours, [r['heating_supply_target_C'] for r in result.hydraulics],
                         label='HK-Vorlaufsoll', alpha=.8, linestyle=':')
        axes[1].plot(hours, [r['heating_return_temperature_C'] for r in result.hydraulics], label='HK-Rücklauf', alpha=.8)
        if result.hydraulics[0].get('heating_return_target_C') is not None:
            axes[1].plot(hours, [r['heating_return_target_C'] for r in result.hydraulics],
                         label='HK-Rücklaufsoll', alpha=.8, linestyle='--')
    axes[1].set_ylabel('Temperatur [°C]')
    axes[1].set_xlabel('Stunden seit Simulationsbeginn (UTC)')
    axes[1].grid(alpha=.25)
    axes[1].legend(ncol=3)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def calculate_daily_temperature_profile(result: SimulationResult, dt_hours: float) -> list[dict]:
    """Average equally weighted steps in dt-sized time-of-day bins from midnight."""
    if not isfinite(dt_hours) or not 0 < dt_hours <= 24:
        raise ValueError('Temperature profile dt_hours must be in (0, 24].')
    if len(result.timestamps) != len(result.room_temperatures_C):
        raise ValueError('Timestamps and room temperatures must have equal lengths.')
    if len(result.timestamps) != len(result.storage_temperatures_C):
        raise ValueError('Timestamps and storage temperatures must have equal lengths.')
    # Configured decimal hour values often approximate whole minutes.
    step_seconds = round(dt_hours * 3600, 6)
    if step_seconds <= 0:
        raise ValueError('Temperature profile step must be at least one microsecond.')
    bins = [[] for _ in range(ceil(86400 / step_seconds))]
    for timestamp, temperature, storage_temperature in zip(
        result.timestamps, result.room_temperatures_C, result.storage_temperatures_C
    ):
        instant = datetime.fromisoformat(timestamp)
        seconds = (instant - instant.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
        target = result.target_room_temperature_C
        bins[int(seconds // step_seconds)].append((
            temperature,
            target - result.night_setback.room_reduction_K(instant),
            target - result.comfort_buffer_K - result.night_setback.room_reduction_K(instant, comfort=True),
            storage_temperature,
        ))
    rows = []
    for index, samples in enumerate(bins):
        row = {'hour': index * step_seconds / 3600, 'sample_count': len(samples)}
        for column, name in enumerate(('room_temperature_C', 'target_room_temperature_C', 'comfort_threshold_C', 'storage_temperature_C')):
            row[name] = sum(sample[column] for sample in samples) / len(samples) if samples else None
        row['room_temperature_min_C'] = min(sample[0] for sample in samples) if samples else None
        row['room_temperature_max_C'] = max(sample[0] for sample in samples) if samples else None
        rows.append(row)
    return rows


def plot_daily_temperature_profile(result: SimulationResult, dt_hours: float,
                                   output_path: str | Path) -> Path:
    """Save the mean daily temperature profile and its aggregated CSV data."""
    rows = calculate_daily_temperature_profile(result, dt_hours)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with plot_csv_path(path).open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.environ.setdefault('MPLCONFIGDIR', str(path.parent))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    x = [row['hour'] for row in rows] + [24]
    lower = [row['room_temperature_min_C'] if row['room_temperature_min_C'] is not None else float('nan') for row in rows]
    upper = [row['room_temperature_max_C'] if row['room_temperature_max_C'] is not None else float('nan') for row in rows]
    ax.fill_between(x, lower + [lower[-1]], upper + [upper[-1]], step='post',
                    color='#1f77b4', alpha=.18, linewidth=0,
                    label='Raumtemperatur (Minimum–Maximum)')
    for key, label, color, style in (
        ('room_temperature_C', 'Raumtemperatur (Mittelwert)', '#1f77b4', '-'),
        ('target_room_temperature_C', 'Solltemperatur', '#e68100', '--'),
        ('comfort_threshold_C', 'Komforttemperatur', '#288348', ':'),
    ):
        values = [row[key] if row[key] is not None else float('nan') for row in rows]
        ax.step(x, values + [values[-1]], where='post', label=label, color=color, linestyle=style)
    ax.set(xlim=(0, 24), xlabel='Uhrzeit', ylabel='Raum-, Soll- und Komforttemperatur [°C]',
           title=f'Durchschnittlicher Tagesverlauf · Zeitraster {dt_hours * 60:g} min')
    ax.set_xticks(range(0, 25, 2), [f'{hour:02d}:00' for hour in range(0, 25, 2)])
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)
    return path


def calculate_daily_heat_profile(result: SimulationResult, dt_hours: float) -> list[dict]:
    """Average actual heat flows in time-of-day bins; losses use step-start temperature."""
    if not isfinite(dt_hours) or not 0 < dt_hours <= 24:
        raise ValueError('Heat profile dt_hours must be in (0, 24].')
    step_seconds = round(dt_hours * 3600, 6)
    if step_seconds <= 0:
        raise ValueError('Heat profile step must be at least one microsecond.')
    names = ('heating_power_to_building_W', 'thermal_power_W', 'building_heat_loss_W', 'storage_temperature_C')
    series = [getattr(result, 'storage_temperatures_C' if name == 'storage_temperature_C' else name)
              for name in names]
    if any(len(values) != len(result.timestamps) for values in series):
        raise ValueError('Timestamps and heat flow series must have equal lengths.')
    rows = [dict(hour=i * step_seconds / 3600, sample_count=0,
                 **{name: 0.0 for name in names}) for i in range(ceil(86400 / step_seconds))]
    for index, timestamp in enumerate(result.timestamps):
        instant = datetime.fromisoformat(timestamp)
        seconds = (instant - instant.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
        row = rows[int(seconds // step_seconds)]
        row['sample_count'] += 1
        for name, values in zip(names, series):
            row[name] += values[index]
    for row in rows:
        for name in names:
            row[name] = row[name] / row['sample_count'] if row['sample_count'] else None
    return rows


def plot_daily_heat_profile(result: SimulationResult, dt_hours: float,
                            output_path: str | Path, *, storage_max_C: float | None = None) -> Path:
    rows = calculate_daily_heat_profile(result, dt_hours)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with plot_csv_path(path).open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.environ.setdefault('MPLCONFIGDIR', str(path.parent))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    x = [row['hour'] for row in rows] + [24]
    for key, label, color, style in (
        ('heating_power_to_building_W', 'Wärmeabgabe ans Gebäude', '#1f77b4', '-'),
        ('thermal_power_W', 'Wärmeleistung der Wärmepumpe', '#e68100', '--'),
        ('building_heat_loss_W', 'Wärmeverlust des Gebäudes', '#288348', ':'),
    ):
        values = [row[key] / 1000 if row[key] is not None else float('nan') for row in rows]
        ax.step(x, values + [values[-1]], where='post', label=label, color=color, linestyle=style)
    ax.set(xlim=(0, 24), xlabel='Uhrzeit', ylabel='Mittlere Wärmeleistung [kW]',
           title=f'Durchschnittliche Wärmeleistungen · Zeitraster {dt_hours * 60:g} min')
    ax.set_xticks(range(0, 25, 2), [f'{hour:02d}:00' for hour in range(0, 25, 2)])
    ax.axhline(0, color='0.5', linewidth=.6)
    ax.grid(alpha=.25)
    storage_ax = ax.twinx()
    temperatures = [row['storage_temperature_C'] if row['storage_temperature_C'] is not None else float('nan') for row in rows]
    storage_ax.step(x, temperatures + [temperatures[-1]], where='post', color='#9467bd',
                label='Puffertemperatur (Mittelwert)')
    storage_ax.set_ylabel('Mittlere Puffertemperatur [°C]', color='#9467bd')
    if storage_max_C is not None:
        storage_ax.set_ylim(0, storage_max_C)
    storage_ax.tick_params(axis='y', colors='#9467bd')
    storage_ax.spines['right'].set_color('#9467bd')
    lines, labels = ax.get_legend_handles_labels()
    storage_lines, storage_labels = storage_ax.get_legend_handles_labels()
    ax.legend(lines + storage_lines, labels + storage_labels)
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)
    return path




def calculate_daily_power_shares(result: SimulationResult, dt_hours: float) -> list[float | None]:
    """Mean fraction of available full power per time-of-day bin, including off steps."""
    if not isfinite(dt_hours) or not 0 < dt_hours <= 24:
        raise ValueError('Power profile dt_hours must be in (0, 24].')
    step_seconds = round(dt_hours * 3600, 6)
    if step_seconds <= 0:
        raise ValueError('Power profile step must be at least one microsecond.')
    fractions = (getattr(result, 'heat_pump_relative_thermal_powers', []) or
                 [max(0.0, value) if value == -1 else value for value in result.heat_pump_power_fractions])
    if len(result.timestamps) != len(fractions):
        raise ValueError('Timestamps and power fractions must have equal lengths.')
    count = ceil(86400 / step_seconds)
    totals, sums = [0] * count, [0.0] * count
    for timestamp, fraction in zip(result.timestamps, fractions):
        if not isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('Power fraction must be finite and in [0, 1].')
        instant = datetime.fromisoformat(timestamp)
        seconds = (instant - instant.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
        index = int(seconds // step_seconds)
        totals[index] += 1
        sums[index] += fraction
    return [value / total if total else None for value, total in zip(sums, totals)]


def plot_hourly_runtime_shares(result: SimulationResult, output_path: str | Path,
                               dt_hours: float) -> Path:
    """Save mean modulated power by time of day; retain the legacy output filename."""
    shares = calculate_daily_power_shares(result, dt_hours)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    width = round(dt_hours * 3600, 6) / 3600
    bins = [index for index, share in enumerate(shares) if share is not None]
    ax.bar([index * width for index in bins], [100 * shares[index] for index in bins],
           width=[min(width, 24 - index * width) for index in bins], align='edge', color="#1f77b4")
    missing = [index for index, share in enumerate(shares) if share is None]
    if missing:
        ax.bar([index * width for index in missing], [100] * len(missing),
               width=[min(width, 24 - index * width) for index in missing], align='edge', color="none", edgecolor="0.8",
               hatch="///", label="Keine Daten")
        ax.legend()
    ax.set_xticks(range(0, 25, 2), [f"{hour:02d}:00" for hour in range(0, 25, 2)])
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Uhrzeit")
    ax.set_ylabel("Mittlere WP-Leistung [% der verfügbaren Vollleistung]")
    ax.set_title(f"Wärmepumpe: mittlerer Tagesverlauf · Zeitraster {dt_hours * 60:g} min")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)
    return path


def write_simple_plot_data(result: SimulationResult, output_path: str | Path,
                           output_steps: int = 1) -> None:
    """Placeholder for future plotting export.

    For now this just writes a compact CSV-friendly text file so the
    project already has a single place to hook plots in later.
    """
    path = Path(output_path)
    path.write_text(
        "\n".join(
            [
                "timestamp,room_temperature_C,storage_temperature_C,outside_temperature_C,cop",
                *[
                    f"{result.timestamps[i]},{result.room_temperatures_C[i]:.3f},{result.storage_temperatures_C[i]:.3f},{result.outside_temperatures_C[i]:.3f},{result.cop[i]:.3f}"
                    for i in range(0, len(result.timestamps), output_steps)
                ],
            ]
        ),
        encoding="utf-8",
    )


def plot_building_temperatures(result: SimulationResult, dt_hours: float, output_path: Path):
    """Keep air and mass histories visible instead of averaging them together."""
    import numpy as np
    import matplotlib.pyplot as plt
    hours = (np.arange(len(result.timestamps)) + 1) * dt_hours
    fig, ax = plt.subplots(figsize=(12, 4), layout='constrained')
    ax.plot(hours, result.room_temperatures_C, label='Raumluft')
    if result.mass_temperatures_C and result.mass_temperatures_C[0] is not None:
        ax.plot(hours, result.mass_temperatures_C, label='Gebaeudemasse')
    ax.set(xlabel='Zeit nach Einschwingphase [h]', ylabel='Temperatur [C]')
    ax.legend()
    ax.grid(alpha=.3)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
