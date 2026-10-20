"""Abbildung rl_validierung.pdf: Validierungsscore beider RL-Regelungen ueber die Trainingsentscheidungen.
Start im Projektordner: python tools/plot_rl_validierung.py [Ausgabe.pdf]"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = [('Leistungsstufe', 'results/rl', '#1f6fb4'), ('Heizkurvenverschiebung', 'results/rl_offset', '#d1792a')]

def latest(folder):
    """Trainingsordner des aktuellen model.zip: 'training' (Stand der Arbeit), sonst das neueste training_*."""
    chosen = json.loads((ROOT / folder / 'model.json').read_text())['selected_checkpoint_timesteps']
    for d in [ROOT / folder / 'training'] + sorted((ROOT / folder).glob('training_*'), reverse=True):
        # Nur abgeschlossene Trainings (report.json); abgebrochene Laeufe ueberspringen
        if (d / 'evaluations.npz').is_file() and (d / 'report.json').is_file():
            return d, chosen

out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / 'Doc/References/rl_validierung.pdf'
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for label, folder, color in RUNS:
    d, chosen = latest(folder)
    data = np.load(d / 'evaluations.npz')
    t, score = data['timesteps'], -data['results'][:, 0]
    ax.plot(t / 1000, score, marker='o', ms=3, lw=1.4, color=color, label=label)
    i = int(np.argmin(np.abs(t - chosen)))
    ax.plot(t[i] / 1000, score[i], 'o', ms=8, mfc='none', mec='red', mew=1.6)
    ax.annotate(f'{chosen // 1000} 000', (t[i] / 1000, score[i]), xytext=(6, 8), textcoords='offset points', fontsize=8, color='red')
ax.set_yscale('log')
ax.set_xlabel('Trainingsentscheidungen [Tausend]')
ax.set_ylabel('Validierungsscore [CHF]')
ax.grid(True, which='both', alpha=0.3)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(out)
print('geschrieben:', out)
