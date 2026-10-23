"""Pipeline: ingest -> extract -> forecast -> health -> brief, plus publishing and scoring.

  python pipeline.py run                      # latest saved pull, rule extractor
  python pipeline.py run --fetch              # new X pull first (costs X API credits)
  python pipeline.py run --extractor gemini-3.8-flash
  python pipeline.py publish                  # latest run -> history/runs (what the dashboard reads)
  python pipeline.py publish 20261023T050000Z 20261024T060000Z
  python pipeline.py score                    # score published runs against history/outcomes.csv

Every run writes data/runs/<run_id>/ (local, gitignored): manifest.json, tagged.jsonl,
forecast.json, health.json and brief.md. Publishing copies only manifest, forecast,
health and scores into history/runs/, which is committed; post text never leaves data/.

outcomes.csv has one row per major escalation: hotspot,date (YYYY-MM-DD),source,note.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import fetch_recent as x
import forecast as fc

RUNS = x.ROOT / "data" / "runs"
HISTORY = x.ROOT / "history"
PUBLISHED_FILES = ("manifest.json", "forecast.json", "health.json", "scores.json")
CODE_FILES = ("fetch_recent.py", "forecast.py", "extract_llm.py", "pipeline.py")


def code_hash() -> str:
    digest = hashlib.sha256()
    for name in CODE_FILES:
        path = x.ROOT / name
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def ingest(fetch: bool) -> dict:
    if fetch:
        collected = x.collect(x.bearer_token(), x.QUERY, x.TARGET)
        return x.save_run(collected, x.QUERY, x.TARGET)
    return x.latest_summary()


def extract(posts: list, extractor: str) -> tuple[list, dict]:
    if extractor == "rules":
        return [fc.classify(post) for post in posts], {"extractor": "rules"}
    import extract_llm

    tagged, stats = extract_llm.tag_posts(posts, model=extractor)
    return tagged, {"extractor": "llm", **stats}


# Phrases that introduce a locality name, used to find places missing from the gazetteer.
PLACE_CUES = r"(?:village|localite|groupement|chefferie|territoire|cite|secteur|axe|near|village of|town of|localities? of) (?:de |d'|du |of )?([a-z][a-z\-]{3,})"


def health(summary: dict, tagged: list, result: dict, previous: dict | None) -> dict:
    """Checks that should stop a forecast from going out unreviewed."""
    known = {alias for _, _, aliases in fc.HOTSPOTS.values() for alias in aliases}
    gaps = Counter()
    for row in tagged:
        if not row["on_topic"] or row["historical"]:
            continue
        for place in re.findall(PLACE_CUES, fc.normalize(row["text"])):
            if place not in known:
                gaps[place] += 1
        for place in row.get("other_places") or []:
            gaps[fc.normalize(place)] += 1
    usable_share = result["usable_posts"] / max(len(tagged), 1)
    errors = sum(1 for row in tagged if row.get("extraction_error"))
    checks = {
        "posts_returned": summary["returned"],
        "usable_share": round(usable_share, 3),
        "languages": summary["languages"],
        "extraction_errors": errors,
        "gazetteer_gaps": gaps.most_common(10),
    }
    warnings = []
    if summary.get("stopped_early"):
        warnings.append(f"X pull stopped early: {summary['stopped_early']}")
    if summary["returned"] < 0.8 * summary["requested"]:
        warnings.append(f"only {summary['returned']} of {summary['requested']} posts returned")
    if usable_share < 0.5:
        warnings.append(f"usable share {usable_share:.0%} is below 50%")
    if errors > 0.02 * len(tagged):
        warnings.append(f"{errors} extraction errors")
    if previous:
        before = previous["health"]["usable_share"]
        if abs(usable_share - before) > 0.2:
            warnings.append(f"usable share moved from {before:.0%} to {usable_share:.0%} since the last run")
        old_top = {row["hotspot"] for row in previous["top5"]}
        new_top = {row["hotspot"] for row in result["rows"][:5]}
        if len(old_top & new_top) <= 2:
            warnings.append(f"top-5 hotspots changed a lot: new {sorted(new_top - old_top)}")
    checks["warnings"] = warnings
    return checks


def brief(manifest: dict, result: dict, checks: dict) -> str:
    """One-page markdown for readers who do not build models."""
    lines = [
        f"# DRC escalation watch, issued {manifest['issued']}",
        "",
        f"Source: {manifest['counts']['posts']} posts on X from {manifest['window'][0]} to {manifest['window'][1]}, "
        f"{result['usable_posts']} used after removing off-topic and historical posts. Extractor: {manifest['extractor']['extractor']}"
        + (f" ({manifest['extractor'].get('model')})" if manifest["extractor"].get("model") else "") + ".",
        "",
        "Probability of a major escalation (10+ deaths in one event, a locality changing hands, or 5,000+ newly displaced). "
        "These numbers are not yet calibrated against outcomes; read them as a ranking with direction.",
        "",
        "| Hotspot | 30 days | 60 days | 90 days | Change vs baseline (30d) | Signal |",
        "|---|---|---|---|---|---|",
    ]
    for row in result["rows"][:8]:
        signal = f"{row['event']} reporting events, {row['control_change']} control change, {row['escalation']} escalation, {row['deescalation']} de-escalation"
        lines.append(f"| {row['hotspot']} | {row['p30']:.0%} | {row['p60']:.0%} | {row['p90']:.0%} | {row['p30_lift'] * 100:+.0f} pts | {signal} |")
    lines += ["", "## Evidence for the largest changes", ""]
    for row in sorted(result["rows"], key=lambda row: row["p30_lift"], reverse=True)[:3]:
        lines.append(f"**{row['hotspot']}** (actors mentioned: {', '.join(row['actors']) or 'none'})")
        for post in row["evidence"]:
            lines.append(f"- {post['created_at'][:10]}: {post['text'][:240]}")
        lines.append("")
    lines += ["## Caveats", ""]
    lines += [f"- {warning}" for warning in checks["warnings"]] or ["- No health warnings this run."]
    lines += [
        "- X volume reflects attention as much as violence; partisan accounts on all sides post claims of victory.",
        "- Baseline hazards are priors until replaced with ACLED rates.",
    ]
    if checks["gazetteer_gaps"]:
        lines.append("- Places not in the gazetteer: " + ", ".join(f"{place} ({n})" for place, n in checks["gazetteer_gaps"]))
    return "\n".join(lines) + "\n"


def previous_run(before: str) -> dict | None:
    runs = sorted(path for path in RUNS.glob("*/manifest.json") if path.parent.name < before)
    return json.loads(runs[-1].read_text()) if runs else None


def run(fetch: bool, extractor: str, acled: Path | None, acled_months: float) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RUNS / run_id
    out.mkdir(parents=True)
    timings = {}

    started = time.time()
    summary = ingest(fetch)
    posts = x.read_jsonl(Path(summary["posts_file"]))
    timings["ingest"] = round(time.time() - started, 2)

    started = time.time()
    tagged, extractor_stats = extract(posts, extractor)
    x.write_jsonl(out / "tagged.jsonl", tagged)
    timings["extract"] = round(time.time() - started, 2)

    started = time.time()
    base = fc.base_rates_from_acled(acled, acled_months) if acled else None
    result = fc.forecast(posts, base_hazard=base, tagged=tagged)
    timings["forecast"] = round(time.time() - started, 2)

    days = sorted(summary["posts_per_day"])
    manifest = {
        "run_id": run_id,
        "issued": (date.fromisoformat(days[-1]) + timedelta(days=1)).isoformat(),
        "window": [days[0], days[-1]],
        "code_hash": code_hash(),
        "inputs": {"posts_file": summary["posts_file"], "query": summary["query"], "acled": str(acled) if acled else None},
        "extractor": extractor_stats,
        "base_rates": "acled" if base else "tier priors",
        "counts": {
            "posts": len(posts),
            "usable": result["usable_posts"],
            "dropped_off_topic": result["dropped_off_topic"],
            "dropped_historical": result["dropped_historical"],
        },
        "timings_seconds": timings,
    }
    fc.save_forecast(result, summary, path=out / "forecast.json", run_id=run_id, code_hash=manifest["code_hash"], extractor=extractor_stats)
    checks = health(summary, tagged, result, previous_run(run_id))
    manifest["health"] = {key: checks[key] for key in ("usable_share", "extraction_errors", "warnings")}
    manifest["top5"] = [{"hotspot": row["hotspot"], "p30": row["p30"]} for row in result["rows"][:5]]
    (out / "health.json").write_text(json.dumps(checks, indent=2))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "brief.md").write_text(brief(manifest, result, checks))
    return out


def read_outcomes(path: Path) -> list:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    events = []
    for row in rows:
        hotspot = fc.resolve_hotspot(row["hotspot"])
        if hotspot is None:
            raise ValueError(f"unknown hotspot in outcomes: {row['hotspot']!r}")
        events.append((hotspot, date.fromisoformat(row["date"].strip())))
    return events


def publish(run_ids: list) -> list:
    """Copy runs' post-free files from data/runs into history/runs for the dashboard."""
    if not run_ids:
        local = sorted(path.parent.name for path in RUNS.glob("*/manifest.json"))
        if not local:
            raise SystemExit("no local runs to publish")
        run_ids = local[-1:]
    published = []
    for run_id in run_ids:
        source = RUNS / run_id
        if not (source / "forecast.json").exists():
            raise SystemExit(f"no forecast for run {run_id} in {RUNS}")
        target = HISTORY / "runs" / run_id
        target.mkdir(parents=True, exist_ok=True)
        for name in PUBLISHED_FILES:
            if (source / name).exists():
                (target / name).write_bytes((source / name).read_bytes())
        published.append(target)
    return published


def score(outcomes_path: Path, runs_dir: Path = HISTORY / "runs", today: date | None = None) -> list:
    """Score every run in `runs_dir` whose horizons have passed. Returns one line per run and horizon."""
    today = today or datetime.now(timezone.utc).date()
    events = read_outcomes(outcomes_path)
    lines = []
    for path in sorted(runs_dir.glob("*/forecast.json")):
        saved = json.loads(path.read_text())
        issued = date.fromisoformat(saved["issued"])
        outcomes = {}
        for row in saved["rows"]:
            after = [(day - issued).days for spot, day in events if spot == row["hotspot"] and day >= issued]
            outcomes[row["hotspot"]] = min(after) if after else None
        scores = fc.brier(path, outcomes)
        for horizon, resolve_by in saved["resolve_by"].items():
            if date.fromisoformat(resolve_by) > today:
                continue
            lines.append({"run_id": saved.get("run_id", path.parent.name), "horizon": horizon, **(scores[horizon] or {})})
        (path.parent / "scores.json").write_text(json.dumps({"scored_on": today.isoformat(), "scores": scores}, indent=2))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run")
    go.add_argument("--fetch", action="store_true", help="pull new posts from X first")
    go.add_argument("--extractor", default="rules", help='"rules", a gemini-* or a claude-* model id')
    go.add_argument("--acled", type=Path, help="ACLED CSV export for base rates")
    go.add_argument("--acled-months", type=float, default=12)
    share = sub.add_parser("publish")
    share.add_argument("run_ids", nargs="*", help="local run ids; default is the latest run")
    grade = sub.add_parser("score")
    grade.add_argument("--outcomes", type=Path, default=HISTORY / "outcomes.csv")
    grade.add_argument("--runs", type=Path, default=HISTORY / "runs")
    args = parser.parse_args()

    if args.command == "run":
        out = run(args.fetch, args.extractor, args.acled, args.acled_months)
        manifest = json.loads((out / "manifest.json").read_text())
        print(f"run {manifest['run_id']} -> {out}")
        for warning in manifest["health"]["warnings"]:
            print(f"WARNING: {warning}")
        print((out / "brief.md").read_text().split("## Evidence")[0])
        sys.exit(1 if manifest["health"]["warnings"] and args.fetch else 0)
    elif args.command == "publish":
        for path in publish(args.run_ids):
            print(f"published {path}")
    else:
        if not args.outcomes.exists():
            print(f"no outcomes file at {args.outcomes} yet")
            return
        lines = score(args.outcomes, args.runs)
        for line in lines:
            print(line)
        if not lines:
            print("no run has reached a resolve-by date yet")


if __name__ == "__main__":
    main()
