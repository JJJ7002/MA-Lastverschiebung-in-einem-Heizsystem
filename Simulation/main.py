from pathlib import Path
import csv
import argparse
import yaml
from dataclasses import asdict
from time import perf_counter

from analysis.metrics import calculate_metrics, metric_report
from analysis.scoring import ScoringParams, calculate_score
from analysis.plots import plot_daily_heat_profile
from analysis.plots import plot_storage_stratification
from analysis.plots import plot_hourly_runtime_shares, write_simple_plot_data, plot_daily_temperature_profile
from analysis.visualize import try_plot_heating_curve, plot_heat_pump_maps, plot_cop_load_lift
from analysis.output_paths import plot_csv_path
from simulation.config import load_config
from simulation.engine import build_result_rows, output_interval_steps, run_simulation


def main() -> None:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Run a heat pump simulation')
    parser.add_argument('--config', type=Path, default=project_root / 'config/default.yaml')
    parser.add_argument('--start', help='UTC start, YYYY-MM-DD HH:MM')
    parser.add_argument('--end', help='UTC end, exclusive')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.start:
        config['simulation']['start_time'] = args.start
    if args.end:
        config['simulation']['end_time'] = args.end
    if args.offline:
        config['weather']['offline'] = True
    if args.no_plots:
        config['analysis']['save_plots'] = False
    if args.output:
        config['analysis']['output_dir'] = str(args.output.resolve())
    scoring = ScoringParams.from_config(config['scoring'])
    started = perf_counter()
    last_report = started

    def report_progress(phase, completed, total):
        nonlocal last_report
        now = perf_counter()
        if completed == 0 or completed == total or now - last_report >= 5:
            progress = f': {completed / total:.0%}' if total else ''
            print(f'{phase}{progress} ({now - started:.1f} s)', flush=True)
            last_report = now

    result = run_simulation(config, progress_callback=report_progress)
    output_steps = output_interval_steps(config)
    print('Ergebnisse auswerten und speichern ...', flush=True)
    metrics = calculate_metrics(
        result,
        comfort_buffer_K=float(config["building"]["comfort_buffer_K"]),
        dt_hours=(float(config["simulation"]["dt_minutes"]) / 60),
        target_temperature_C=float(config["building"]["T_set_C"]),
    )

    score = calculate_score(result, metrics, (float(config['simulation']['dt_minutes']) / 60), scoring)
    output_dir = (project_root / config["analysis"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'building_model.yaml').write_text(yaml.safe_dump(dict(parameters=result.building_parameters, initial_temperatures_C=result.initial_building_temperatures_C, settling_hours=result.settling_hours, initial_building_energy_Wh=result.initial_building_energy_Wh, initial_storage_energy_Wh=result.initial_storage_energy_Wh, measurement_initial_state=result.measurement_initial_state), sort_keys=False), encoding='utf-8')
    (output_dir / 'effective_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')

    if bool(config["analysis"]["save_csv"]):
        from simulation.engine import save_settling_csv
        save_settling_csv(result, output_dir / "settling_results.csv", output_steps)
        rows = build_result_rows(result, output_steps)
        csv_path = output_dir / "simulation_results.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with (output_dir / "simulation_metrics.csv").open("w", encoding="utf-8", newline="") as fh:
            summary = metric_report(metrics)
            summary.update({key: value for key, value in asdict(score).items() if key != 'currency'})
            writer = csv.DictWriter(fh, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)

    if bool(config["analysis"]["save_plots"]):
        print('Diagramme erstellen ...', flush=True)
        from analysis.plots import plot_building_temperatures
        plot_building_temperatures(result, float(config['simulation']['dt_minutes']) / 60,
                                  output_dir / 'building_temperatures.png')
        plot_storage_stratification(result, float(config['simulation']['dt_minutes']) / 60,
                                   output_dir / 'storage_stratification.png')
        plot_daily_heat_profile(result, (float(config['simulation']['dt_minutes']) / 60),
                                output_dir / 'daily_heat_profile.png',
                                storage_max_C=float(config['heat_pump']['T_sink_max_C']))
        plot_daily_temperature_profile(result, (float(config['simulation']['dt_minutes']) / 60),
                                       output_dir / 'daily_temperature_profile.png')
        write_simple_plot_data(result, plot_csv_path(output_dir / "plot_data.png"), output_steps)
        plot_hourly_runtime_shares(result, output_dir / "heat_pump_hourly_runtime.png",
                                  (float(config['simulation']['dt_minutes']) / 60))
        try_plot_heating_curve(config, output_dir / "heating_curve.png")
        plot_heat_pump_maps(config, output_dir)
        plot_cop_load_lift(config, output_dir / 'powerSetting_cop_W35.png')
        plot_cop_load_lift(config, output_dir / 'powerSetting_cop_W55.png', 55.0)

    print("Four-port stratified buffer simulation complete.")
    print(f"Project root: {project_root}")
    print(f"Completed {len(result.timestamps)} physical simulation steps.")
    print(f"Seasonal COP: {metrics.seasonal_cop:.2f}")
    print(f"Heat pump starts: {metrics.heat_pump_start_count} ({metrics.heat_pump_starts_per_day:.2f} per day)")
    if config['control'].get('method') == 'cop_optimized':
        print(f"Heating-curve fallback above COP demand: {metrics.fallback_override_hours:.2f} h")
    print(f"Time below scheduled comfort temperature: {metrics.comfort_below_hours:.2f} h")
    print(f"Comfort temperature deficit: {metrics.comfort_degree_hours:.2f} K h")
    print(f"Final room temperature: {result.room_temperatures_C[-1]:.2f} °C")
    print(f"Final mean storage temperature: {result.storage_mean_temperatures_C[-1]:.2f} °C")
    print(f"Final buffer control sensor: {result.storage_temperatures_C[-1]:.2f} °C")
    print(f"Final buffer top/bottom: {result.storage_node_temperatures_C[-1][0]:.2f} / "
          f"{result.storage_node_temperatures_C[-1][-1]:.2f} °C")
    print(f"Total electric energy: {result.total_electric_energy_Wh:.2f} Wh")
    print(f"Total thermal energy: {result.total_thermal_energy_Wh:.2f} Wh")
    if config['control'].get('method') in ('pause_load_shifting', 'run_load_shifting'):
        summary = result.load_shifting_summary
        label = config['control']['method'].replace('_', ' ').capitalize()
        print(f"{label}: {summary['accepted_shifts']} accepted shifts in {summary['interval_trials']} trials")
        print(f"Sum of score gains in accepted trial comparisons: {summary['score_gain']:.4f} {score.currency} "
              f"(electricity {summary['electricity_gain']:.4f}, starts {summary['start_gain']:.4f}, "
              f"comfort {summary['comfort_gain']:.4f})")
        print(f"  saved electricity {summary['saved_electric_energy_Wh']:.2f} Wh, "
              f"{summary['extra_starts']} extra starts, comfort deficit change {summary['comfort_deficit_change_Kh']:.3f} K h")
    print(f"Electricity cost: {score.electricity_cost:.2f} {score.currency}")
    print(f"Cycle penalty: {score.cycle_cost:.2f} {score.currency}")
    print(f"Comfort penalty: {score.comfort_cost:.2f} {score.currency}")
    print(f"Total score (lower is better): {score.total_score:.2f} {score.currency}")
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
