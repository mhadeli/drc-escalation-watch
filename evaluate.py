"""Gold labels and evaluation for the post extractors.

  python evaluate.py sample            # write data/labels/to_label.csv (200 posts, stratified)
  python evaluate.py score             # score the rules extractor against labelled rows
  python evaluate.py score --model claude-haiku-5-5 --model claude-opus-5-5

Labelling: fill the gold columns in to_label.csv, then save it as data/labels/gold.csv.
Booleans are 1/0, lists are separated by ";". Hotspots can be short names ("Walungu",
"Kaziba"); actors use the names in forecast.ACTORS. Leave `relevant` blank for rows you have not labelled yet; they are skipped.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import fetch_recent as x
import forecast as fc

LABELS = x.ROOT / "data" / "labels"
GOLD_COLUMNS = ("relevant", "historical", "event", "control_change", "escalation", "deescalation", "hotspots", "actors", "fatalities")
BOOL_FIELDS = ("relevant", "historical", "event", "control_change", "escalation", "deescalation")
SET_FIELDS = ("hotspots", "actors")


def sample(posts: list, n: int = 200, seed: int = 7) -> list:
    """Stratify by language and by what the rules think, so rare cases are in the set.

    Strata: rule-tagged events, rule-tagged history, posts with a hotspot, and the rest,
    each split by language. Sampling by rule output on purpose over-represents the cases
    that matter; the gold labels themselves must be written without looking at the rules.
    """
    rng = random.Random(seed)
    strata = defaultdict(list)
    for post in posts:
        row = fc.classify(post)
        kind = "event" if row["event"] else "historical" if row["historical"] else "hotspot" if row["hotspots"] else "other"
        lang = post.get("lang") if post.get("lang") in ("fr", "en") else "other"
        strata[(kind, lang)].append(post)
    picked, share = [], n / len(posts)
    for key in sorted(strata):
        group = strata[key]
        rng.shuffle(group)
        picked += group[: max(3, round(len(group) * share))]
    rng.shuffle(picked)
    return picked[:n]


def write_sheet(posts: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("post_id", "created_at", "lang", "text", *GOLD_COLUMNS, "notes"))
        for post in posts:
            text = " ".join((post.get("text") or "").split())
            writer.writerow((post["id"], post.get("created_at", "")[:10], post.get("lang"), text, *[""] * len(GOLD_COLUMNS), ""))


def read_gold(path: Path) -> list:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if (raw.get("relevant") or "").strip() == "":
                continue
            row = {"post_id": raw["post_id"], "lang": raw.get("lang") or "und"}
            for field in BOOL_FIELDS:
                row[field] = (raw.get(field) or "0").strip() in ("1", "true", "yes", "y")
            for field in SET_FIELDS:
                row[field] = {item.strip() for item in (raw.get(field) or "").split(";") if item.strip()}
            row["hotspots"] = {fc.resolve_hotspot(item) or item for item in row["hotspots"]}
            value = (raw.get("fatalities") or "").strip()
            row["fatalities"] = int(value) if value else None
            rows.append(row)
    return rows


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0 if tp + fp + fn else None
    return {"precision": _r(precision), "recall": _r(recall), "f1": _r(f1), "support": tp + fn}


def _r(value):
    return None if value is None else round(value, 3)


def score(gold: list, predicted: dict) -> dict:
    """Field-level metrics. `predicted` maps post id -> tagged row."""
    report = {"n": 0, "fields": {}, "by_language": {}, "errors": []}
    counts = defaultdict(lambda: [0, 0, 0])  # field -> tp, fp, fn
    by_lang = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    fatality_hits, fatality_total = 0, 0
    for gold_row in gold:
        row = predicted.get(gold_row["post_id"])
        if row is None:
            continue
        report["n"] += 1
        pred = {
            "relevant": row["on_topic"],
            "historical": row["historical"],
            **{field: row[field] for field in fc.SIGNALS},
            "hotspots": set(row["hotspots"]),
            "actors": set(row["actors"]),
        }
        lang = gold_row["lang"] if gold_row["lang"] in ("fr", "en") else "other"
        for field in BOOL_FIELDS:
            truth, guess = gold_row[field], bool(pred[field])
            cell = counts[field]
            cell[0] += truth and guess
            cell[1] += guess and not truth
            cell[2] += truth and not guess
            if field == "event":
                lang_cell = by_lang[lang]["event"]
                lang_cell[0] += truth and guess
                lang_cell[1] += guess and not truth
                lang_cell[2] += truth and not guess
        for field in SET_FIELDS:
            truth, guess = gold_row[field], pred[field]
            for cell in (counts[field], by_lang[lang][field]):
                cell[0] += len(truth & guess)
                cell[1] += len(guess - truth)
                cell[2] += len(truth - guess)
            if field == "hotspots" and truth != guess:
                report["errors"].append({"post_id": gold_row["post_id"], "field": field, "gold": sorted(truth), "predicted": sorted(guess)})
        if gold_row["fatalities"] is not None:
            fatality_total += 1
            fatality_hits += row.get("fatalities") == gold_row["fatalities"]
        for field in ("event", "historical"):
            if gold_row[field] != bool(pred[field]):
                report["errors"].append({"post_id": gold_row["post_id"], "field": field, "gold": gold_row[field], "predicted": bool(pred[field])})
    report["fields"] = {field: _prf(*counts[field]) for field in (*BOOL_FIELDS, *SET_FIELDS)}
    report["fields"]["fatalities_exact"] = {"accuracy": _r(fatality_hits / fatality_total) if fatality_total else None, "support": fatality_total}
    report["by_language"] = {lang: {field: _prf(*cell) for field, cell in fields.items()} for lang, fields in sorted(by_lang.items())}
    return report


def predictions(posts: list, extractor: str) -> tuple[dict, dict]:
    if extractor == "rules":
        return {post["id"]: fc.classify(post) for post in posts}, {}
    import extract_llm

    tagged, stats = extract_llm.tag_posts(posts, model=extractor)
    return {row["id"]: row for row in tagged}, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("sample")
    make.add_argument("-n", type=int, default=200)
    run = sub.add_parser("score")
    run.add_argument("--gold", type=Path, default=LABELS / "gold.csv")
    run.add_argument("--model", action="append", default=[], help="LLM model id; repeat to compare. Rules are always scored.")
    args = parser.parse_args()

    posts = x.read_jsonl(Path(x.latest_summary()["posts_file"]))
    if args.command == "sample":
        path = LABELS / "to_label.csv"
        write_sheet(sample(posts, args.n), path)
        print(f"wrote {path}")
        return

    gold = read_gold(args.gold)
    wanted = {row["post_id"] for row in gold}
    subset = [post for post in posts if post["id"] in wanted]
    results = {}
    for extractor in ["rules", *args.model]:
        predicted, stats = predictions(subset, extractor)
        results[extractor] = {**score(gold, predicted), "cost": stats}
    out = LABELS / "eval_report.json"
    out.write_text(json.dumps(results, indent=2))
    for extractor, result in results.items():
        f1 = {field: metrics.get("f1") for field, metrics in result["fields"].items() if "f1" in metrics}
        print(f"{extractor:20} n={result['n']}  " + "  ".join(f"{k}={v}" for k, v in f1.items()))
    print(f"full report: {out}")


if __name__ == "__main__":
    main()
