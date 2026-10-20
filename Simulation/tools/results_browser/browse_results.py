"""Local, dependency-free browser for simulation CSVs. Run from any directory."""
from __future__ import annotations

import argparse
import csv
import errno
import json
import math
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import webbrowser


def discover_runs(root: Path) -> list[dict]:
    runs = []
    for path in sorted(root.rglob('simulation_results.csv')):
        if not path.resolve().is_relative_to(root.resolve()):
            continue
        with path.open(encoding='utf-8-sig', newline='') as stream:
            columns = next(csv.reader(stream), [])
        if 'timestamp' not in columns:
            continue
        relative = path.relative_to(root)
        runs.append(dict(id=relative.as_posix(), name=path.parent.name,
                         study=relative.parent.parent.as_posix(),
                         columns=[column for column in columns if column != 'timestamp']))
    return runs


def load_run(root: Path, run_id: str) -> dict:
    path = (root / run_id).resolve()
    if not path.is_relative_to(root.resolve()) or path.name != 'simulation_results.csv':
        raise ValueError('Ungültiger Run-Pfad')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = [key for key in (reader.fieldnames or []) if key != 'timestamp']
        data = {key: [] for key in columns}
        times = []
        for row in reader:
            # Treat naive simulation times as wall-clock UTC to avoid browser DST shifts.
            instant = datetime.fromisoformat(row['timestamp'])
            if instant.tzinfo is None:
                from datetime import timezone
                instant = instant.replace(tzinfo=timezone.utc)
            times.append(instant.timestamp() * 1000)
            for key in columns:
                value = row.get(key, '')
                try:
                    number = float({'True': '1', 'False': '0'}.get(value, value))
                    data[key].append(number if math.isfinite(number) else None)
                except (ValueError, TypeError):
                    data[key].append(None)
    return dict(times=times, data=data)


def run_directory(root: Path, run_id: str) -> Path:
    path = (root / run_id).resolve()
    if not path.is_relative_to(root.resolve()) or path.name != 'simulation_results.csv':
        raise ValueError('Ungültiger Run-Pfad')
    return path.parent


def attribution(root: Path, query: dict) -> dict:
    """Split the electricity difference of two runs into causes (needs the project's code and config.yaml)."""
    project = str(Path(__file__).resolve().parents[2])
    if project not in sys.path:
        sys.path.insert(0, project)
    from analysis.efficiency_attribution import compute
    first = lambda key: query.get(key, [''])[0] or None
    base, compare = run_directory(root, first('base') or ''), run_directory(root, first('compare') or '')
    for folder in (base, compare):
        if not (folder / 'config.yaml').is_file():
            raise ValueError(f'config.yaml fehlt in {folder.name}')
    return compute(base, compare, first('from'), first('to'))


def handler_for(root: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlparse(self.path)
            try:
                if request.path == '/':
                    body = Path(__file__).with_name('results_browser.html').read_bytes()
                    content_type = 'text/html; charset=utf-8'
                elif request.path == '/api/runs':
                    body = json.dumps(discover_runs(root)).encode()
                    content_type = 'application/json'
                elif request.path == '/api/run':
                    run_id = parse_qs(request.query).get('id', [''])[0]
                    body = json.dumps(load_run(root, run_id), allow_nan=False).encode()
                    content_type = 'application/json'
                elif request.path == '/api/attribution':
                    body = json.dumps(attribution(root, parse_qs(request.query)), allow_nan=False).encode()
                    content_type = 'application/json'
                else:
                    self.send_error(404)
                    return
            except (OSError, ValueError, KeyError, ImportError) as exc:
                self.send_error(400, str(exc))
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_results = Path(__file__).resolve().parents[2] / 'results'
    parser.add_argument('--results', type=Path, default=default_results)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    root = args.results.resolve()
    if not root.is_dir():
        parser.error(f'Results directory does not exist: {root}')
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(root))
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(root))
        print(f'Port {args.port} ist belegt; verwende Port {server.server_port}.', flush=True)
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'Simulationsbrowser: {url}\nDaten: {root}\nBeenden mit Strg+C', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
