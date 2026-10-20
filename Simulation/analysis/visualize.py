from __future__ import annotations

from pathlib import Path
from typing import Any
import os

from models.heat_pump import get_performance
from analysis.output_paths import plot_csv_path


def calculate_radiator_heating_curve(config: dict[str, Any]) -> list[dict[str, float]]:
    """Stationary demand at the room setpoint, mapped to radiator temperatures."""
    from models.building import BuildingParams
    from models.radiator import RadiatorMap
    from control.temperature_targets import RadiatorHeatingCurve

    building = BuildingParams.from_config(config['building'])
    radiator = RadiatorMap(volume_flow_L_h=float(config['heating']['volume_flow_L_h']),
                           **config['heating']['radiator'])
    room = float(config['building']['T_set_C'])
    settings = config['control'].get('heating_curve', {})
    curve = RadiatorHeatingCurve(radiator, building.stationary_UA_W_per_K, room,
                                float(settings.get('slope_factor', 1)),
                                float(settings.get('offset_K', 0)),
                                float(settings.get('reference_outside_C', 20)))
    rows = []
    for outside in range(-20, 26):
        demand = max(0.0, building.stationary_UA_W_per_K * (room - outside))
        supply, ret = radiator.temperatures(demand)
        rows.append(dict(outside_temp_C=float(outside), room_target_C=room,
                         heating_demand_W=demand, heating_supply_target_C=room + supply,
                         heating_return_target_C=room + ret,
                         buffer_target_C=curve.target_temperature_C(outside)))
    return rows


def _plot_radiator_heating_curve(config, output_path):
    import csv
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = calculate_radiator_heating_curve(config)
    csv_path = plot_csv_path(path)
    with csv_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        os.environ.setdefault('MPLCONFIGDIR', str(path.parent))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return csv_path
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = [r['outside_temp_C'] for r in rows]
    ax.plot(xs, [r['heating_supply_target_C'] for r in rows],
            label='Benötigter Heizkreisvorlauf', linewidth=2.2)
    ax.plot(xs, [r['heating_return_target_C'] for r in rows],
            label='Zugehöriger Heizkreisrücklauf', linestyle='--', linewidth=2)
    ax.plot(xs, [r['buffer_target_C'] for r in rows],
            label='Angepasste Puffer-Heizkurve (vor WP-Begrenzung)',
            linestyle=':', color='black', linewidth=2)
    room = rows[0]['room_target_C']
    ax.set(xlabel='Außentemperatur [°C]', ylabel='Heizkreistemperatur [°C]',
           xlim=(xs[0], xs[-1]),
           title=f'Bedarfsbasierte Heizkurve · Raumsoll {room:g} °C\n'
                 'Stationärer Wärmebedarf und Heizkörperkennfeld')
    ax.text(.02, .03, 'Ab Außentemperatur = Raumsoll: kein Heizbedarf',
            transform=ax.transAxes, fontsize=9)
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)
    return path


def try_plot_heating_curve(config: dict[str, Any], output_path: str | Path) -> Path:
    """Plot the demand-based radiator curve and export its underlying CSV."""
    return _plot_radiator_heating_curve(config, output_path)


def _plot_heatmap(x, y, values, output, *, title, xlabel, ylabel, label,
                  cmap='viridis', operating_lines=None):
    """Draw only the coloured surface, including a curved minimum-output boundary."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 7), layout='constrained')
    surface = ax.pcolormesh(x, y, values, shading='gouraud', cmap=cmap,
                            vmin=float(values.min()), vmax=float(values.max()),
                            rasterized=True)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    if operating_lines is not None:
        lifts, minimum, optimal = operating_lines
        ax.plot(lifts, minimum, color='#d62728', linewidth=2,
                label='Mindestleistung im Dauerbetrieb')
        ax.plot(lifts, optimal, color='black', linewidth=4)
        ax.plot(lifts, optimal, color='white', linewidth=2)
        # A black legend handle remains visible on the white legend background.
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([], [], color='#d62728', linewidth=2, label='Mindestleistung im Dauerbetrieb'),
            Line2D([], [], color='black', linewidth=3, label='Effizienteste Leistungsstufe (max. COP)'),
        ], loc='best')
    fig.colorbar(surface, ax=ax, label=label, pad=.03)
    try:
        fig.savefig(output, dpi=170)
    finally:
        plt.close(fig)


def plot_heat_pump_maps(config: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Plot COP at maximum output and maximum heat output using the simulation model."""
    from simulation.engine import _build_heat_pump_params
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    params = _build_heat_pump_params(config)
    source_min = max(-20.0, params.source_temp_min_C)
    sink_max = min(65.0, params.sink_temp_max_C)
    outside = np.arange(source_min, 30.01, 1.0)
    storage = np.arange(30.0, sink_max + .01, 1.0)
    x, y = np.meshgrid(outside, storage)
    cop = np.zeros_like(x)
    power = np.zeros_like(x)
    rows = []
    for i in range(len(storage)):
        for j in range(len(outside)):
            cop[i,j], thermal, _ = get_performance(float(x[i,j]), float(y[i,j]), params)
            power[i,j] = thermal / 1000
            rows.append((x[i,j], y[i,j], cop[i,j], thermal))
    with plot_csv_path(path / "heat_pump_maps.png").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["outside_temp_C", "storage_temp_C", "cop_at_max_power", "thermal_max_W"])
        writer.writerows(rows)

    outputs = []
    for values, name, title, label, cmap in (
        (cop, "cop_map.png", "COP bei maximaler Heizleistung", "COP [-]", "viridis"),
        (power, "max_power_map.png", "Maximale Heizleistung", "Heizleistung [kW]", "plasma"),
    ):
        fig = plt.figure(figsize=(11, 8), layout="constrained")
        ax = fig.add_subplot(111, projection="3d")
        surface = ax.plot_surface(x, y, values, cmap=cmap, linewidth=.2,
                                  edgecolor=(0,0,0,.12), antialiased=True,
                                  rcount=len(storage), ccount=len(outside))
        ax.set_xlabel("Außentemperatur [°C]", labelpad=12)
        ax.set_ylabel("WP-Vorlauftemperatur [°C]", labelpad=12)
        ax.set_zlabel(label, labelpad=12)
        ax.set_title(title + "\nWPL 25 A · WP-Vorlauftemperatur", pad=20)
        ax.view_init(elev=28, azim=-130)
        ax.set_box_aspect((1.3, 1, .8))
        fig.colorbar(surface, ax=ax, shrink=.65, pad=.08, label=label)
        output = path / name
        try:
            fig.savefig(output, dpi=170)
        finally:
            plt.close(fig)
        outputs.append(output)
        _plot_heatmap(x, y, values, output.with_name(output.stem + '_2d.png'),
                      title=title + '\nWPL 25 A · WP-Vorlauftemperatur',
                      xlabel='Außentemperatur [°C]', ylabel='WP-Vorlauftemperatur [°C]',
                      label=label, cmap=cmap)
    # Remove only superseded plot artifacts in the selected output directory.
    for name in ("cop_curve.png", "cop_curve_temperature_lift.png",
                 "cop_curve.csv", "cop_curve_temperature_lift.csv"):
        (path / name).unlink(missing_ok=True)
    return tuple(outputs)


def plot_cop_load_lift(config: dict[str, Any], output_path: str | Path,
                      storage_temp_C: float = 35.0) -> Path:
    """COP between minimum and maximum continuous output at fixed storage temperature."""
    from simulation.engine import _build_heat_pump_params
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    params = _build_heat_pump_params(config)
    storage_temp_C = min(storage_temp_C, params.sink_temp_max_C)
    if storage_temp_C < 30:
        raise ValueError('Storage temperature must be in the mapped range, 30–65 °C')
    sources = np.arange(max(-20., params.source_temp_min_C), 30.01)
    lifts = storage_temp_C - sources
    levels = np.linspace(0., 1., 39)
    x = np.tile(lifts, (len(levels), 1))
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    rows = []
    minimum_fractions, minimum_cops = [], []
    optimal_fractions, optimal_cops, optimal_rows = [], [], []
    for j, source in enumerate(sources):
        curve = params.performance_map.curves(float(source), storage_temp_C)
        minimum_fraction = curve[0][0]/curve[-1][0]
        minimum_fractions.append(100*minimum_fraction)
        minimum_cops.append(curve[0][1])
        fractions = np.linspace(minimum_fraction, 1., len(levels))
        for i, fraction in enumerate(fractions):
            command = (fraction - minimum_fraction) / (1 - minimum_fraction)
            cop, power, electric = get_performance(float(source), storage_temp_C, params, max(0.0, float(command)))
            y[i,j] = 100*fraction
            z[i,j] = cop
            rows.append((source, storage_temp_C, lifts[j], fraction, max(0.0, float(command)), cop, power, electric,
                         False))
        # Evaluate every exported load knot: COP is linear between these knots,
        # so a maximum occurs at a knot. Ties select the lower thermal output.
        evaluated = [(q, get_performance(float(source), storage_temp_C, params,
                                        (q - curve[0][0]) / (curve[-1][0] - curve[0][0]))[0]) for q, _ in curve]
        optimal_power, optimal_cop = max(evaluated, key=lambda point: (point[1], -point[0]))
        optimal_fraction = optimal_power / curve[-1][0]
        optimal_fractions.append(100*optimal_fraction)
        optimal_cops.append(optimal_cop)
        optimal_rows.append((source, storage_temp_C, lifts[j], optimal_fraction,
                             (optimal_power - curve[0][0]) / (curve[-1][0] - curve[0][0]),
                             optimal_cop, optimal_power))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with plot_csv_path(path).open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['outside_temp_C','storage_temp_C','temperature_lift_K',
                         'relative_thermal_power','power_command','cop','thermal_power_W','electric_power_W','below_minimum_output'])
        writer.writerows(rows)
    with plot_csv_path(path, suffix='_optimal.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['outside_temp_C','storage_temp_C','temperature_lift_K',
                         'optimal_relative_thermal_power','optimal_power_command','maximum_cop','thermal_power_W'])
        writer.writerows(optimal_rows)
    fig = plt.figure(figsize=(11,8), layout='constrained')
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(x,y,z,cmap='viridis',edgecolor=(0,0,0,.12),
                              linewidth=.2,rcount=len(levels),ccount=len(sources))
    ax.plot(lifts,minimum_fractions,minimum_cops,color='#d62728',linewidth=2.5,
            label='Mindestleistung im Dauerbetrieb')
    # Draw a contrasting outline so the optimum remains visible on the surface.
    ax.plot(lifts,optimal_fractions,optimal_cops,color='black',linewidth=4,
            label='Effizienteste Leistungsstufe (max. COP)',zorder=10)
    ax.plot(lifts,optimal_fractions,optimal_cops,color='white',linewidth=2,
            marker='o',markersize=2.5,
            zorder=11)
    ax.set_xlabel('Temperaturhub WP-Vorlauf − Außenluft [K]',labelpad=12)
    ax.set_xlim(float(max(lifts)), float(min(lifts)))
    ax.set_ylabel('Heizleistungsstufe [% vom Maximum]',labelpad=12)
    ax.set_ylim(100., float(min(minimum_fractions)))
    ax.set_zlabel('COP [-]',labelpad=10)
    ax.set_title(f'COP über Leistungsstufe und Temperaturhub\nWPL 25 A · WP-Vorlauftemperatur {storage_temp_C:g} °C',pad=20)
    ax.view_init(elev=28,azim=-130)
    ax.set_box_aspect((1.3,1,.8))
    ax.legend(loc='upper left')
    fig.colorbar(surface,ax=ax,shrink=.65,pad=.08,label='COP [-]')
    fig.supxlabel('Dargestellt ist ausschließlich der Dauerbetrieb zwischen Mindest- und Maximalleistung.',fontsize=10)
    try:
        fig.savefig(path,dpi=170)
    finally:
        plt.close(fig)
    _plot_heatmap(x, y, z, path.with_name(path.stem + '_2d.png'),
                  title=f'COP über Leistungsstufe und Temperaturhub\nWPL 25 A · WP-Vorlauftemperatur {storage_temp_C:g} °C',
                  xlabel='Temperaturhub WP-Vorlauf − Außenluft [K]',
                  ylabel='Heizleistungsstufe [% vom Maximum]', label='COP [-]',
                  operating_lines=(lifts, minimum_fractions, optimal_fractions))
    return path
