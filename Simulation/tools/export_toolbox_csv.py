"""Normalize the captured chart tables (data/wpl25a_toolbox_raw.json -> data/wpl25a_toolbox.csv);
never fetch or invent manufacturer data."""
import csv
import json
import re
from pathlib import Path


def number(text):
    return float(re.search(r"-?\d+(?:\.\d+)?", text).group())


def main():
    root = Path(__file__).resolve().parents[1] / "data"
    raw = json.loads((root / "wpl25a_toolbox_raw.json").read_text(encoding="utf-8"))
    fields = ["source_temp_C", "sink_temp_C", "inverter_speed_percent",
              "thermal_min_W", "thermal_max_W", "thermal_power_W", "cop",
              "electric_min_W", "electric_max_W", "electric_power_chart_W"]
    with (root / "wpl25a_toolbox.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(fields)
        for table in raw["tables"]:
            for heat, cop, electric in zip(table["heat"][1:], table["cop"][1:], table["electric"][1:], strict=True):
                assert heat[0] == cop[0] == electric[0]
                writer.writerow([number(heat[0]), table["sink_temp_C"], table["inverter_speed_percent"],
                                 *[number(v) * 1000 for v in heat[1:4]], number(cop[1]),
                                 *[number(v) * 1000 for v in electric[1:4]]])


if __name__ == "__main__":
    main()
