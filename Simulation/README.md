# Simulation einer Wärmepumpenheizung mit Lastverschiebung

Python-Simulation zur Maturaarbeit. Sie bildet Gebäude, Heizkörper und Heizkreis, einen geschichteten
Pufferspeicher (Stiebel Eltron SBP 700 E) und eine Luft-Wasser-Wärmepumpe (Stiebel Eltron WPL 25 A) im
Zeitverlauf ab und vergleicht damit verschiedene Regelstrategien: zwei Heizkurvenregelungen, zwei Arten
der Lastverschiebung, eine COP-Regelung und zwei Regler mit Reinforcement Learning (RL).

Die Ergebnisse der Arbeit liegen in `results/` und lassen sich mit den Befehlen unten nachrechnen.

## Einrichten

Benötigt wird **Python 3.12** (unter <https://www.python.org/downloads/>; bei der Installation unter
Windows «Add python.exe to PATH» anwählen). Mit neueren Python-Versionen funktioniert die Simulation zwar,
die RL-Bibliothek PyTorch ist dafür aber unter Umständen noch nicht verfügbar.

1. Repository herunterladen: auf GitHub «Code → Download ZIP» und entpacken, oder
   `git clone https://github.com/JJJ7002/MA-Lastverschiebung-in-einem-Heizsystem.git`
2. Eine Eingabeaufforderung oder PowerShell im Ordner `Simulation` öffnen.
3. Eine eigene Python-Umgebung anlegen und die Pakete installieren:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\activate
   python -m pip install -r requirements-rl.txt
   ```

   `requirements-rl.txt` enthält alles, auch die RL-Pakete (PyTorch ist ein grosser Download). Wer die
   RL-Regler nicht braucht, installiert nur `requirements.txt`.
4. Prüfen, ob alles funktioniert:

   ```powershell
   python -m pytest
   ```

Auf macOS und Linux funktioniert alles gleich, nur wird die Umgebung mit `source .venv/bin/activate`
aktiviert, und die `.cmd`-Startdateien entfallen.

Alle Befehle werden im Ordner `Simulation` mit aktivierter Umgebung ausgeführt. Beim ersten Lauf lädt
die Simulation die Wetterdaten von MeteoSchweiz herunter (Internet nötig) und speichert sie in
`weather_cache/`.

## Ordnerstruktur

```
main.py              eine einzelne Simulation
run_experiments.py   mehrere Simulationen mit veränderten Parametern (Vergleiche, Studien)
train_rl.py          Training der beiden RL-Regler
run_simulation.cmd   Windows: main.py mit der Umgebung .venv starten

config/      alle Einstellungen: Anlage (default.yaml), Studien, RL-Training, Validierung
models/      physikalische Modelle: Gebäude, Heizkörper, Hydraulik, Speicher, Wärmepumpe
control/     Regelstrategien
simulation/  Ablauf der Simulation und Laden der Konfiguration
learning/    Trainingsumgebung und Training der RL-Regler
weather/     Download und Einlesen der Wetterdaten
analysis/    Kennzahlen, Total Score, Diagramme
data/        Eingangsdaten: Wärmepumpenkennfeld, Speicherdaten, Wetterdaten für die Tests
tools/       Abbildungen der Arbeit, Ergebnisbrowser, Hilfsskripte
tests/       automatisierte Tests
results/     Ergebnisse der Arbeit und die trainierten RL-Modelle
```

## Benutzung

### Einstellungen

Alle Einstellungen der Anlage stehen in `config/default.yaml`; jeder Eintrag ist dort kommentiert.
Die wichtigsten:

- `simulation.start_time` / `end_time`: Zeitraum (UTC, wie die Wetterdaten)
- `control.method`: Regelstrategie, eine von `heating_curve`, `heating_curve_supply`, `cop_optimized`,
  `pause_load_shifting`, `run_load_shifting`, `reinforcement_learning`, `reinforcement_learning_offset`
- `scoring`: Preise und Strafkosten des Total Score

Für eigene Versuche am besten eine Kopie anlegen (z. B. `config/mein_versuch.yaml`) und diese mit
`--config` angeben.

### Eine Simulation

```powershell
python main.py                                   # mit config/default.yaml
python main.py --config config/mein_versuch.yaml
python main.py --start "2026-01-01 00:00" --end "2026-02-01 00:00"
```

Ergebnis: `results/single_run/` mit Zeitreihe (`simulation_results.csv`), Kennzahlen
(`simulation_metrics.csv`), verwendeter Konfiguration und Diagrammen. Dieser Ordner wird bei jedem Lauf
überschrieben; mit `--output <Ordner>` wird anderswo gespeichert.

### Mehrere Simulationen vergleichen

Eine Studie ist eine YAML-Datei mit einer Basiskonfiguration und den Parametern, die verändert werden.
Gerechnet werden alle Kombinationen, dazu die unveränderte Basis als `run_000`. Beispiel
(`config/experiments.yaml`, der Hauptvergleich der Arbeit):

```yaml
base_config: default.yaml
output_dir: ../results/experiments
save_time_series: true
parameters:
  control.method: [heating_curve_supply, pause_load_shifting, run_load_shifting,
                   reinforcement_learning, reinforcement_learning_offset]
```

Die Schlüssel unter `parameters` sind Pfade in `default.yaml`, z. B. `control.heating_curve.offset_K: [6, 7, 8]`.

```powershell
python run_experiments.py                                          # config/experiments.yaml
python run_experiments.py --config config/validation/01_zeitschritt.yaml
```

`--workers auto` (Standard) rechnet mehrere Läufe parallel. Ergebnis: ein neuer Ordner
`study_<Datum>_<Zeit>` im angegebenen `output_dir` mit einem Unterordner pro Lauf sowie `summary.csv`
(alle Kennzahlen) und `ranking.csv` (nach Total Score sortiert).

### RL-Regler

Die beiden trainierten Modelle liegen in `results/rl/` und `results/rl_offset/` und werden mit
`control.method: reinforcement_learning` bzw. `reinforcement_learning_offset` ohne weiteres Training
verwendet. Neu trainieren:

```powershell
python train_rl.py                                            # Regler mit Leistungsstufen
python train_rl.py --config config/rl_training_offset.yaml    # Regler mit Heizkurvenverschiebung
```

Ein Training mit den Einstellungen der Arbeit kann mehrere Stunden dauern. Es legt
`results/rl/training_<Datum>_<Zeit>/` an und **überschreibt `results/rl/model.zip`** (bzw. `rl_offset`).
Weil das Training nicht auf jedem Rechner gleich abläuft, entsteht dabei ein etwas anderes Modell als in
der Arbeit; das Original lässt sich mit Git zurückholen (`git checkout -- results/rl`).

### Ergebnisse ansehen

```powershell
tools\results_browser\start_results_browser.cmd
# oder: python tools/results_browser/browse_results.py [--results <Ordner>]
```

Der Browser öffnet sich unter <http://127.0.0.1:8765>. Er durchsucht `results/` mit allen Unterordnern
und zeigt jeden Lauf mit Zeitreihe, gruppiert nach Ordner (z. B. `hauptvergleich`, `experiments/study_…`;
`single_run` erscheint als «Einzelruns»). Das Suchfeld filtert nach Ordner- und Laufnamen; mit `--results` lässt sich ein bestimmter
Ordner wählen. Läufe ohne Zeitreihe erscheinen nicht; das betrifft die Validierungsläufe, die nur
Kennzahlen speichern (`save_time_series: false` in ihrer Konfiguration).

### Wohin neue Läufe geschrieben werden

| Befehl | Ausgabe |
|---|---|
| `python main.py` | `results/single_run/` (wird überschrieben) |
| `python run_experiments.py` | `results/experiments/study_<Zeit>/` |
| `python run_experiments.py --config config/validation/<datei>.yaml` | `results/validation/<name>/study_<Zeit>/` |
| `python train_rl.py` | `results/rl/training_<Zeit>/`, überschreibt `results/rl/model.zip` |

Neue Läufe werden von Git ignoriert; die Ergebnisse der Arbeit bleiben unverändert.

## Ergebnisse der Arbeit

| Ordner | Inhalt | Konfiguration |
|---|---|---|
| `results/hauptvergleich/` | Vergleich der Regelstrategien im Messjahr 1.8.2025–1.8.2026 (Kapitel Resultate), mit Zeitreihen | `config/experiments.yaml` |
| `results/validation/00_abstimmung_2024_25/` | Abstimmung der Offsets beider Heizkurvenregelungen | `config/validation/abstimmung_2024_25_*.yaml` |
| `results/validation/00_abstimmung_2024_25_nachbarn/` | Nachbarprüfung der gewählten Einstellungen (Anhang; nur zusammengefasste Tabellen) | `config/validation/00_abstimmung_2024_25_nachbarn_*.yaml` |
| `results/validation/01_zeitschritt/` | Konvergenz des Rechenschritts | `config/validation/01_zeitschritt.yaml` |
| `results/validation/03_verlustanteil/` | Sensitivität auf den direkten Verlustanteil | `config/validation/03_verlustanteil.yaml` |
| `results/validation/04_toleranzen/` | Sensitivität der Lastverschiebung auf die Vergleichstoleranz | `config/validation/04_toleranzen_*.yaml` |
| `results/rl/`, `results/rl_offset/` | trainierte RL-Modelle (`model.zip`, `model.json`) und ihr Training (`training/`) | `config/rl_training*.yaml` |

Läufe im Hauptvergleich: `run_000` `heating_curve` (Referenz), `run_001` `heating_curve_supply`,
`run_002` `pause_load_shifting`, `run_003` `run_load_shifting`, `run_004` `reinforcement_learning`,
`run_005` `reinforcement_learning_offset`. Jeder Lauf enthält seine vollständige Konfiguration
(`config.yaml`). In `01_zeitschritt` liegen zusätzlich Läufe mit `pause_load_shifting`, die in der
Arbeit nicht verwendet werden.

### Nachrechnen

```powershell
python run_experiments.py                                                    # Hauptvergleich
python run_experiments.py --config config/validation/01_zeitschritt.yaml     # analog für die übrigen Validierungen
python tools/resultate_abbildungen.py --out abbildungen                      # Abbildungen und Tabellen der Resultate
python tools/plot_rl_validierung.py abbildungen/rl_validierung.pdf           # Validierungsverlauf der RL-Trainings
```

`resultate_abbildungen.py` verwendet ohne Angabe `results/hauptvergleich` und die Validierungen in
`results/validation`; wurde eine Validierung neu gerechnet, nimmt es deren neuesten `study_`-Ordner.
Einen eigenen Hauptvergleich gibt man als Ordner an:
`python tools/resultate_abbildungen.py results/experiments/study_<Zeit> --out abbildungen`.

## Wetterdaten

Standardmässig lädt die Simulation das Archiv der MeteoSchweiz-Station Basel-Binningen herunter
(10-Minuten-Werte, UTC). Mit `weather.offline: true` in der Konfiguration oder `python main.py --offline`
werden nur bereits heruntergeladene Dateien verwendet. Die Wetterdateien in `data/weather/` dienen den
Tests.

## Datenquellen

- Wetterdaten: MeteoSchweiz, Open Government Data, Station Basel-Binningen
- Wärmepumpenkennfeld (`data/wpl25a_toolbox.csv`) und Speicherdaten (`data/sbp_e_models.yaml`):
  Herstellerunterlagen von Stiebel Eltron
