import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Thread
from http.server import ThreadingHTTPServer
from urllib.request import urlopen
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from browse_results import discover_runs, load_run, handler_for


class BrowserTests(unittest.TestCase):
    def test_discovery_conversion_and_http(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'study_one' / 'run_000'
            run.mkdir(parents=True)
            (run / 'simulation_results.csv').write_text(
                'timestamp,room_temperature_C,heat_pump_on,cop\n'
                '2020-01-20 12:00:00,21,False,\n'
                '2020-01-20 13:00:00,22,True,nan\n', encoding='utf-8')
            runs = discover_runs(root)
            self.assertEqual(runs[0]['study'], 'study_one')
            data = load_run(root, runs[0]['id'])
            self.assertEqual(data['data']['heat_pump_on'], [0, 1])
            self.assertEqual(data['data']['cop'], [None, None])
            self.assertEqual(data['times'][1] - data['times'][0], 3600000)
            for path in ('../simulation_results.csv', str(root.parent / 'simulation_results.csv')):
                with self.assertRaises(ValueError):
                    load_run(root, path)
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(root))
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f'http://127.0.0.1:{server.server_port}'
                with urlopen(base + '/api/runs') as response:
                    self.assertEqual(json.load(response), runs)
                with urlopen(base + '/api/run?id=' + runs[0]['id']) as response:
                    self.assertEqual(json.load(response), data)
                with urlopen(base) as response:
                    self.assertIn('Simulationsbrowser', response.read().decode())
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == '__main__':
    unittest.main()
