"""Run a parameter grid and an unchanged baseline sequentially."""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, fields
from datetime import datetime, timedelta
from itertools import chain, product
from pathlib import Path

import yaml

from analysis.metrics import calculate_metrics, metric_report, metric_report_fields
from analysis.plots import plot_daily_temperature_profile
from analysis.plots import plot_daily_heat_profile
from analysis.scoring import Score, ScoringParams, calculate_score
from simulation.config import load_config
from weather.meteoswiss import download_files, weather_source, load_api_weather, load_api_daily, parse_time
from simulation.engine import build_result_rows, output_interval_steps, run_simulation


def parameter_location(config: dict, path: str):
    if not isinstance(path, str) or not path or any(not key for key in path.split('.')):
        raise ValueError(f'Invalid parameter path: {path}')
    value = config
    for key in path.split('.'):
        parent = value
        if isinstance(parent, list) and key.isdecimal():
            key = int(key)
            if key >= len(parent):
                raise ValueError(f'Unknown parameter: {path}')
        elif not isinstance(parent, dict) or key not in parent:
            raise ValueError(f"Unknown parameter: {path}")
        value = parent[key]
    return parent, key


def parameter_value(config: dict, path: str):
    parent, key = parameter_location(config, path)
    value = parent[key]
    if isinstance(value, (dict, list)):
        raise ValueError(f"Parameter must name a scalar setting: {path}")
    return value


def configurations(base: dict, parameters: dict):
    """Validate first; yield independent baseline and grid configurations."""
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError('parameters must be a nonempty mapping')
    for path, values in parameters.items():
        parameter_value(base, path)
        if path.startswith('analysis.'):
            raise ValueError('analysis.* output settings are managed by the experiment runner')
        if not isinstance(values, list) or not values:
            raise ValueError(f'Expected a nonempty value list for {path}')
        if any(isinstance(value, (dict, list)) for value in values):
            raise ValueError(f'Expected scalar parameter values for {path}')
    yield 'baseline', deepcopy(base)
    for values in product(*parameters.values()):
        config = deepcopy(base)
        for path, value in zip(parameters, values):
            target, key = parameter_location(config, path)
            target[key] = deepcopy(value)
        yield 'variant', config


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _init_worker() -> None:
    # Worker processes only write plot files; never open a GUI backend.
    import os
    os.environ.setdefault('MPLBACKEND', 'Agg')


def _resolve_workers(workers, case_count: int) -> int:
    if workers in (None, 'auto', 0):
        workers = os.cpu_count() or 1
    workers = int(workers)
    if workers < 1:
        raise ValueError('workers must be a positive integer or "auto"')
    return max(1, min(workers, case_count))


def _run_case(job: dict) -> dict:
    """Run, score and write one case. Returns the summary row (picklable)."""
    row, config, run_dir, settings = job['row'], job['config'], job['run_dir'], job['settings']
    result = None
    try:
        if job['weather_error']:
            raise _PreparedError(job['weather_error'])
        scoring = ScoringParams.from_config(config['scoring'])
        result = run_simulation(config, **job['weather_inputs'])
        (run_dir / 'building_model.yaml').write_text(yaml.safe_dump(dict(parameters=result.building_parameters, initial_temperatures_C=result.initial_building_temperatures_C, settling_hours=result.settling_hours, initial_building_energy_Wh=result.initial_building_energy_Wh, initial_storage_energy_Wh=result.initial_storage_energy_Wh, measurement_initial_state=result.measurement_initial_state), sort_keys=False), encoding='utf-8')
        metrics = calculate_metrics(
            result, comfort_buffer_K=float(config['building']['comfort_buffer_K']),
            dt_hours=(float(config['simulation']['dt_minutes']) / 60),
            target_temperature_C=float(config['building']['T_set_C']),
        )
        score = calculate_score(result, metrics, (float(config['simulation']['dt_minutes']) / 60), scoring)
        row.update(metric_report(metrics))
        row.update({key: value for key, value in asdict(score).items() if key != 'currency'})
        row.update(trial_counts(result.load_shifting_summary))
        if settings.get('save_heat_profile', False):
            plot_daily_heat_profile(result, (float(config['simulation']['dt_minutes']) / 60),
                                   run_dir / 'daily_heat_profile.png',
                                   storage_max_C=float(config['heat_pump']['T_sink_max_C']))
        if settings.get('save_temperature_profile', False):
            from analysis.plots import plot_building_temperatures
            plot_building_temperatures(result, float(config['simulation']['dt_minutes']) / 60,
                                      run_dir / 'building_temperatures.png')
            plot_daily_temperature_profile(result, (float(config['simulation']['dt_minutes']) / 60),
                                           run_dir / 'daily_temperature_profile.png')
    except Exception as exc:
        # A broken variant must not discard the rest of the study.
        message = exc.args[0] if isinstance(exc, _PreparedError) else f'{type(exc).__name__}: {exc}'
        row.update(status='error', error=message)
        (run_dir / 'error.txt').write_text(row['error'], encoding='utf-8')
        return dict(row=row, completed=False)
    write_csv(run_dir / 'simulation_metrics.csv', [row])
    if config['analysis']['save_csv']:
        from simulation.engine import save_settling_csv
        output_steps = output_interval_steps(config)
        save_settling_csv(result, run_dir / 'settling_results.csv', output_steps)
        write_csv(run_dir / 'simulation_results.csv', build_result_rows(result, output_steps))
    return dict(row=row, completed=True)


TRIAL_COUNT_FIELDS = ['trials_total', 'trials_accepted', 'trials_no_saving', 'trials_no_recovery', 'trials_infeasible']


def trial_counts(summary: dict) -> dict:
    """Outcome counts of the paired trials of the load-shifting strategies (empty for other methods).

    trials_no_recovery: comparisons that did not return to the reference state within the
    comparison horizon (outcomes 'timeout' and 'no_recovery').
    """
    trials = (summary or {}).get('trials')
    if not trials:
        return {}
    outcomes = [t.get('outcome') for t in trials]
    return dict(trials_total=len(outcomes),
                trials_accepted=outcomes.count('accepted'),
                trials_no_saving=outcomes.count('no_saving'),
                trials_no_recovery=outcomes.count('timeout') + outcomes.count('no_recovery'),
                trials_infeasible=outcomes.count('infeasible'))


class _PreparedError(Exception):
    """Weather preparation failed earlier; carries the original message."""


def run_experiments(experiment_path: Path, workers=1) -> Path:
    """Run the study. workers: 1 = sequential (default), N, or "auto" (one per CPU core).

    Cases are independent, so parallel execution gives the same results; summary.csv
    and ranking.csv are still written in run order.
    """
    experiment_path = experiment_path.resolve()
    settings = load_config(experiment_path)
    base = load_config(experiment_path.parent / settings['base_config'])
    parameters = settings['parameters']
    cases = configurations(base, parameters)
    first = next(cases)  # Validate before creating output directories.
    output_root = (experiment_path.parent / settings['output_dir']).resolve()
    study_dir = output_root / datetime.now().strftime('study_%Y-%m-%d_%H%M%S_%f')
    study_dir.mkdir(parents=True, exist_ok=False)
    (study_dir / 'experiments.yaml').write_text(yaml.safe_dump(settings, sort_keys=False), encoding='utf-8')
    columns = ['run_id', 'kind', 'status', 'error', *parameters,
               *metric_report_fields(),
               *[field.name for field in fields(Score) if field.name != 'currency'],
               *TRIAL_COUNT_FIELDS]
    completed = []
    total = 1
    for values in parameters.values():
        total *= len(values)
    total += 1
    worker_count = _resolve_workers(workers, total)
    # Variations of dates/control parameters share one archive per study.
    prepared_weather = {}
    weather_windows = {}
    daily_weather = {}
    project_root = Path(__file__).resolve().parent

    def prepare_case(index, kind, config):
        """Create the run directory and gather the shared weather inputs of one case."""
        run_id = f'run_{index:03d}'
        run_dir = study_dir / run_id
        run_dir.mkdir()
        config['analysis']['output_dir'] = str(run_dir)
        config['analysis']['save_csv'] = bool(settings.get('save_time_series', True))
        config['analysis']['save_plots'] = False
        (run_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        row = dict(run_id=run_id, kind=kind, status='completed', error='')
        row.update({path: parameter_value(config, path) for path in parameters})
        weather_inputs, weather_error = {}, ''
        try:
            if weather_source(config) == 'meteoswiss':
                weather = config.get('weather', {})
                key = (str(weather.get('station', 'BAS')).strip().upper(),
                       str((project_root / weather.get('cache_dir', 'weather_cache')).resolve()),
                       bool(weather.get('offline', False)))
                if key not in prepared_weather:
                    try:
                        prepared_weather[key] = download_files(config, project_root, None)
                    except Exception as exc:
                        prepared_weather[key] = exc
                if isinstance(prepared_weather[key], Exception):
                    raise prepared_weather[key]
                start_time = parse_time(config['simulation'].get('start_time'))
                settling_hours = float(config['simulation'].get('settling_hours', 0))
                settling_start = (start_time - timedelta(hours=settling_hours)
                                  if start_time is not None else None)
                window = (key, settling_start,
                          parse_time(config['simulation'].get('end_time')))
                if window not in weather_windows:
                    weather_config = dict(
                        config,
                        simulation=dict(config['simulation'], start_time=settling_start),
                    )
                    weather_windows[window] = load_api_weather(
                        weather_config, project_root, files=prepared_weather[key])
                weather_inputs['weather_points'] = weather_windows[window]
                if config['control'].get('method') == 'cop_optimized':
                    if key not in daily_weather:
                        daily_weather[key] = load_api_daily(
                            config, project_root, files=prepared_weather[key])
                    weather_inputs['daily_temperatures'] = daily_weather[key]
        except Exception as exc:
            weather_inputs, weather_error = {}, f'{type(exc).__name__}: {exc}'
        job_settings = {name: settings.get(name, False) for name in ('save_heat_profile', 'save_temperature_profile')}
        return dict(row=row, config=config, run_dir=run_dir, settings=job_settings,
                    weather_inputs=weather_inputs, weather_error=weather_error)

    def outcomes():
        jobs = ((index, kind, config) for index, (kind, config) in enumerate(chain([first], cases)))
        if worker_count == 1:
            for index, kind, config in jobs:
                yield _run_case(prepare_case(index, kind, config))
            return
        # Warm the numba cache once so the workers do not all compile it.
        try:
            from models import _tank_kernels
            _tank_kernels.warm_up()
        except Exception:
            pass
        with ProcessPoolExecutor(max_workers=worker_count, initializer=_init_worker) as pool:
            futures = [pool.submit(_run_case, prepare_case(index, kind, config))
                       for index, kind, config in jobs]
            for future in futures:  # In run order, so output stays deterministic.
                yield future.result()

    with (study_dir / 'summary.csv').open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for index, outcome in enumerate(outcomes()):
            row = outcome['row']
            if outcome['completed']:
                completed.append(row)
            writer.writerow(row)
            fh.flush()
            # Only completed, scored runs participate in the ranking.
            ranking = sorted(completed, key=lambda item: (item['total_score'], item['run_id']))
            with (study_dir / 'ranking.csv').open('w', encoding='utf-8', newline='') as ranking_fh:
                ranking_writer = csv.DictWriter(ranking_fh, fieldnames=['rank', *columns])
                ranking_writer.writeheader()
                ranking_writer.writerows(dict(rank=rank, **item) for rank, item in enumerate(ranking, 1))
            print(f'[{index + 1}/{total}] {row["run_id"]} ({row["kind"]}): {row["status"]}', flush=True)
    print(f'Results written to: {study_dir}')
    if completed:
        best = min(completed, key=lambda item: (item['total_score'], item['run_id']))
        print(f"Lowest score: {best['run_id']} ({best['total_score']:.2f})")
    return study_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).resolve().parent / 'config/experiments.yaml')
    parser.add_argument('--workers', default='auto',
                        help='parallel processes: a number, or "auto" for one per CPU core (default); 1 = sequential')
    args = parser.parse_args()
    run_experiments(args.config, workers=args.workers)


if __name__ == '__main__':
    main()
