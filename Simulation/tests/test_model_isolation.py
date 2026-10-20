"""Each simulation must import and read only its own application files."""
from pathlib import Path
import subprocess
import sys


def test_standalone_runtime_uses_local_modules_and_paths():
    root = Path(__file__).resolve().parents[1]
    code = r'''
from pathlib import Path
from datetime import datetime
import sys
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from simulation.config import load_config
from weather.loader import WeatherPoint
from simulation.engine import run_simulation
c = load_config(root / 'config/default.yaml')
c['simulation'].update(start_time=None, end_time=None, settling_hours=0)
run_simulation(c, weather_points=[WeatherPoint(datetime(2025, 1, 1, 12), 0)])
for module in tuple(sys.modules.values()):
    name = getattr(module, '__file__', None)
    if name and getattr(module, '__name__', '').split('.')[0] in ('models', 'control', 'simulation', 'weather', 'learning', 'analysis'):
        assert Path(name).resolve().is_relative_to(root), name
for setting in [c['heat_pump']['performance_map_file'], c['weather']['cache_dir'], c['analysis']['output_dir']]:
    assert (root / setting).resolve().is_relative_to(root), setting
'''
    result = subprocess.run([sys.executable, '-I', '-c', code, str(root)],
                            cwd=root, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
