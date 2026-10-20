from pathlib import Path


def plot_csv_path(plot_path: str | Path, *, suffix: str = '.csv') -> Path:
    """Return the CSV companion path inside the result folder's ``csv`` directory."""
    plot_path = Path(plot_path)
    csv_dir = plot_path.parent / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir / f'{plot_path.stem}{suffix}'
