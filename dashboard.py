"""Build the static dashboard (site/index.html) from saved runs.

  python dashboard.py                                   # local runs in data/runs
  python dashboard.py --runs history/runs --out site    # what CI does with the data branch

Only manifests, forecasts, health checks and scores are read, so the page never
contains post text and is safe to publish on GitHub Pages.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import forecast as fc

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "dashboard" / "index.html"

# Approximate centres (lat, lon) for the map; territory-level, not exact event locations.
COORDS = {
    "Walungu (Kaziba, Muhungu, Kamanyola)": (-2.75, 28.75),
    "Uvira (Ruzizi plain)": (-3.10, 29.10),
    "Fizi (Minembwe, Hauts Plateaux)": (-3.85, 28.90),
    "Mwenga (Itombwe, Kamituga)": (-3.10, 28.35),
    "Kalehe (Minova, Numbi)": (-1.95, 28.95),
    "Kabare / Bukavu": (-2.50, 28.86),
    "Shabunda": (-2.69, 27.34),
    "Goma / Nyiragongo": (-1.68, 29.22),
    "Masisi": (-1.40, 28.82),
    "Rutshuru": (-1.19, 29.45),
    "Walikale": (-1.42, 28.05),
    "Lubero / Butembo": (0.00, 29.15),
    "Beni": (0.49, 29.47),
    "Djugu": (1.92, 30.50),
    "Irumu / Bunia": (1.56, 30.25),
    "Mambasa": (1.36, 29.05),
    "Mahagi / Aru": (2.60, 30.90),
    "Tanganyika": (-5.94, 29.19),
    "Kwamouth (Mobondo)": (-3.18, 16.22),
    "Kinshasa (political violence)": (-4.32, 15.31),
}
assert set(COORDS) == set(fc.HOTSPOTS), "every hotspot needs map coordinates"


def _read(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def collect(runs_dir: Path) -> list:
    runs = []
    for folder in sorted(runs_dir.glob("*/")):
        saved = _read(folder / "forecast.json")
        manifest = _read(folder / "manifest.json")
        if not saved or not manifest:
            continue
        health = _read(folder / "health.json") or {}
        scores = (_read(folder / "scores.json") or {}).get("scores") or {}
        runs.append({
            "run_id": manifest["run_id"],
            "issued": saved["issued"],
            "window": manifest.get("window"),
            "resolve_by": saved.get("resolve_by"),
            "extractor": manifest.get("extractor", {}),
            "base_rates": manifest.get("base_rates"),
            "counts": manifest.get("counts", {}),
            "health": {
                "usable_share": health.get("usable_share"),
                "extraction_errors": health.get("extraction_errors"),
                "warnings": health.get("warnings", []),
                "gazetteer_gaps": health.get("gazetteer_gaps", []),
            },
            "scores": scores,
            "rows": [
                {key: row.get(key) for key in (
                    "hotspot", "province", "tier", "signal_multiplier", "p30", "p60", "p90",
                    "p30_baseline", "p60_baseline", "p90_baseline", "event", "control_change",
                    "escalation", "deescalation", "authors", "actors")}
                for row in saved["rows"]
            ],
        })
    return runs


def build(runs_dir: Path, out: Path) -> Path:
    payload = {
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "target": "Major escalation: a single event with 10+ deaths, a locality changing hands, or 5,000+ newly displaced.",
        "coords": COORDS,
        "runs": collect(runs_dir),
    }
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    out.mkdir(parents=True, exist_ok=True)
    page = out / "index.html"
    page.write_text(TEMPLATE.read_text().replace("/*__DATA__*/null", data))
    (out / "data.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    return page


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=ROOT / "data" / "runs")
    parser.add_argument("--out", type=Path, default=ROOT / "site")
    args = parser.parse_args()
    page = build(args.runs, args.out)
    count = len(json.loads((args.out / "data.json").read_text())["runs"])
    print(f"{count} runs -> {page}")


if __name__ == "__main__":
    main()
