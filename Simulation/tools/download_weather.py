"""Optional cache preparation; simulation starts download weather automatically."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

from simulation.config import load_config
from weather.meteoswiss import download_files


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description='Vollständige Wetterdaten der konfigurierten Station laden.')
    parser.add_argument('--config', default=str(root / 'config/default.yaml'))
    args = parser.parse_args()
    config = load_config(args.config)
    for path in download_files(config, root, None):
        print(path)
    print('Fertig. Quelle: MeteoSchweiz. Originale UTC-Zeitstempel bleiben erhalten.')


if __name__ == '__main__':
    main()
