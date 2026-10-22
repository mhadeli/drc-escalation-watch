"""Turn a saved X pull into 30/60/90-day escalation forecasts per DRC hotspot.

What is forecast: the probability that a hotspot sees a *major escalation* within
the horizon, meaning at least one of
  - a single attack or clash with 10 or more reported deaths,
  - a town or locality changing hands between armed actors,
  - a new displacement wave of 5,000+ people.
"Any violence" is not used as the target: in most eastern territories it is close
to certain every month, so it would not separate one place from another.

Model: each hotspot has a baseline monthly hazard (BASE_HAZARD, an editable prior
until replaced with ACLED counts via `base_rates_from_acled`). Signals from the
posts multiply that hazard, and the boost decays with time constant SIGNAL_DECAY_DAYS,
so a week of chatter moves the 30-day number much more than the 90-day one.

The numbers are uncalibrated until scored against outcomes. `save_forecast` keeps
every run with its resolve-by dates, and `brier` scores a run once outcomes are known.
"""

import csv
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FORECASTS = ROOT / "data" / "forecasts"
HORIZONS = (30, 60, 90)
SIGNAL_DECAY_DAYS = 21

# Monthly hazard of a major escalation, by tier. Priors, not measurements.
TIER_HAZARD = {"front": 0.5, "elevated": 0.25, "watch": 0.08}

# hotspot -> (province, tier, aliases). Aliases are matched as whole words, accents stripped.
HOTSPOTS = {
    "Walungu (Kaziba, Muhungu, Kamanyola)": ("South Kivu", "front", ("walungu", "kaziba", "muhungu", "munanira", "bideka", "kamanyola", "nyangezi")),
    "Uvira (Ruzizi plain)": ("South Kivu", "front", ("uvira", "luvungi", "katogota", "sange", "kiliba", "luberizi")),
    "Fizi (Minembwe, Hauts Plateaux)": ("South Kivu", "front", ("fizi", "minembwe", "kabanja", "baraka", "bibokoboko")),
    "Mwenga (Itombwe, Kamituga)": ("South Kivu", "elevated", ("mwenga", "itombwe", "kamituga", "mikenge")),
    "Kalehe (Minova, Numbi)": ("South Kivu", "elevated", ("kalehe", "minova", "numbi", "nyabibwe", "lumbishi")),
    "Kabare / Bukavu": ("South Kivu", "elevated", ("kabare", "kavumu", "katana", "bukavu")),
    "Shabunda": ("South Kivu", "watch", ("shabunda",)),
    "Goma / Nyiragongo": ("North Kivu", "elevated", ("goma", "nyiragongo")),
    "Masisi": ("North Kivu", "front", ("masisi", "sake", "kitchanga", "mweso", "ngungu", "nyabiondo")),
    "Rutshuru": ("North Kivu", "front", ("rutshuru", "bwito", "kiwanja", "bunagana", "kanyabayonga", "tongo")),
    "Walikale": ("North Kivu", "front", ("walikale", "pinga", "mpofi")),
    "Lubero / Butembo": ("North Kivu", "front", ("lubero", "butembo", "kirumba", "kipese", "alimbongo")),
    "Beni": ("North Kivu", "front", ("beni", "oicha", "eringeti", "mbau", "kamango", "mangina", "maboya")),
    "Djugu": ("Ituri", "front", ("djugu", "pimbo", "fataki", "lodda", "bule", "savo", "largu", "nizi", "drodro")),
    "Irumu / Bunia": ("Ituri", "front", ("irumu", "komanda", "bunia")),
    "Mambasa": ("Ituri", "elevated", ("mambasa", "biakato")),
    "Mahagi / Aru": ("Ituri", "watch", ("mahagi", "aru")),
    "Tanganyika": ("Tanganyika", "watch", ("tanganyika", "kalemie", "nyunzu", "moba")),
    "Kwamouth (Mobondo)": ("Mai-Ndombe", "watch", ("kwamouth", "mobondo")),
    "Kinshasa (political violence)": ("Kinshasa", "watch", ("kinshasa",)),
}

ACTORS = {
    "M23/AFC": ("m23", "afc"),
    "FRN": ("frn",),
    "RDF (Rwanda)": ("rdf",),
    "FARDC": ("fardc",),
    "Wazalendo": ("wazalendo",),
    "ADF": ("adf",),
    "CODECO": ("codeco",),
    "FDLR": ("fdlr",),
    "Twirwaneho": ("twirwaneho",),
    "Burundi (FDNB)": ("fdnb", "burundais", "burundian"),
    "MONUSCO": ("monusco",),
}

# Each class is a list of regex fragments, applied to lowercased, accent-stripped text.
SIGNALS = {
    "event": (
        r"affrontement", r"\bcombats?\b", r"\bclash", r"fighting", r"\battaque", r"\battack",
        r"\btue(e|s|es)?\b", r"\bkilled\b", r"\bdeaths?\b", r"\bmorts?\b", r"massacre",
        r"bombard", r"\bshell", r"\bdrone", r"enlev", r"abduct", r"kidnap", r"\botages?\b",
        r"deplace", r"displac", r"fuite", r"\bfled\b", r"incendi", r"\bburn",
    ),
    "control_change": (
        r"repris le controle", r"reprennent le controle", r"reprise (du|de|des)", r"\breprennent\b",
        r"liber(e|es|ation)\b", r"\btombe(e)?\b", r"\bcaptur", r"\bseiz", r"\boccupe", r"\btook control",
        r"\bfell to\b", r"\bse rapprochent\b", r"\bavancee", r"\badvanc",
    ),
    "escalation": (
        r"prepar\w* (une|l')\s?offensive", r"\boffensive\b", r"renforts?", r"reinforcement",
        r"mobilis", r"distribu\w* (d'|des )?armes", r"\barmes\b.*distribu", r"\bcoalition\b", r"\balliance\b",
        r"ultimatum", r"encercl", r"\bmenace", r"\bthreat", r"recrut", r"recruit", r"\bescalat",
        r"\bsurrounded\b", r"\bmassing\b", r"prendre les armes",
    ),
    "deescalation": (
        r"cessez-le-feu", r"ceasefire", r"cease-fire", r"\btreve\b", r"truce", r"accord de paix",
        r"peace (deal|agreement|talks)", r"negociation", r"\bnegotiat", r"\bdoha\b", r"\bretrait\b",
        r"withdraw", r"desarmement", r"disarm", r"dialogue",
    ),
}

# Posts about past events (anniversaries, history threads) say little about the next 90 days.
HISTORICAL = (
    r"\b(19[5-9]\d|200\d|201\d|202[0-4])\b.*\b(attaque|attack|massacre|tue|killed|guerre|war)",
    r"\bans apres\b", r"\byears ago\b", r"\bil y a \d+ ans\b", r"anniversaire", r"commemor",
    r"\bsouvenir\b", r"\bmemorial\b", r"\bhommage\b",
)

DRC_CONTEXT = ("kivu", "ituri", "rdc", "congo", "fardc", "m23", "wazalendo", "codeco", "adf", "frn", "kinshasa")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"https?://\S+", " ", text.lower())
    return text.replace("’", "'")


def _any(patterns, text: str) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _words(text: str) -> set:
    """Whole words, plus the parts of hyphenated ones ("frn-m23-rdf" -> frn, m23, rdf)."""
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)
    return set(tokens) | {part for token in tokens for part in token.split("-")}


def classify(post: dict) -> dict:
    """Tag one post with hotspots, actors and signal classes."""
    text = normalize(post.get("text"))
    words = _words(text)
    hotspots = [
        name for name, (_, _, aliases) in HOTSPOTS.items()
        if any(alias in words for alias in aliases)
    ]
    actors = [name for name, keys in ACTORS.items() if any(key in words for key in keys)]
    tags = {name: _any(patterns, text) for name, patterns in SIGNALS.items()}
    on_topic = bool(hotspots) or any(key in words for key in DRC_CONTEXT)
    return {
        "id": post.get("id"),
        "author_id": post.get("author_id"),
        "created_at": post.get("created_at"),
        "text": " ".join((post.get("text") or "").split()),
        "fingerprint": re.sub(r"[^a-z]", "", text)[:120],
        "on_topic": on_topic,
        "historical": _any(HISTORICAL, text),
        "hotspots": hotspots,
        "actors": actors,
        **tags,
    }


def resolve_hotspot(name: str) -> str | None:
    """Full hotspot name from either the name itself or any alias ("Kaziba" -> Walungu)."""
    if name in HOTSPOTS:
        return name
    words = _words(normalize(name))
    for hotspot, (_, _, aliases) in HOTSPOTS.items():
        if words & set(aliases) or words & _words(normalize(hotspot.split("(")[0])):
            return hotspot
    return None


def hotspot_signals(tagged: list) -> dict:
    """Aggregate tagged posts (from `classify` or the LLM extractor) into per-hotspot signal counts.

    Counts are unique authors, and copy-pasted texts count once, so one account
    or one campaign posting many times does not dominate.
    """
    usable = [row for row in tagged if row["on_topic"] and not row["historical"]]
    latest = max((row["created_at"] for row in usable if row["created_at"]), default="")
    cutoff = ""
    if latest:
        cutoff = (datetime.fromisoformat(latest.replace("Z", "+00:00")) - timedelta(hours=48)).isoformat()

    stats = {name: defaultdict(set) for name in HOTSPOTS}
    seen = defaultdict(set)
    for row in usable:
        for name in row["hotspots"]:
            if row["fingerprint"] in seen[name]:
                continue
            seen[name].add(row["fingerprint"])
            spot = stats[name]
            spot["authors"].add(row["author_id"])
            spot["posts"].add(row["id"])
            for signal in SIGNALS:
                if row[signal]:
                    spot[signal].add(row["author_id"])
            if row["event"] and (row["created_at"] or "") >= cutoff.replace("+00:00", ""):
                spot["event_last_48h"].add(row["author_id"])
            for actor in row["actors"]:
                spot["actors"].add(actor)

    by_id = {row["id"]: row for row in usable}
    return {
        "tagged": tagged,
        "usable": len(usable),
        "dropped_off_topic": sum(not row["on_topic"] for row in tagged),
        "dropped_historical": sum(row["on_topic"] and row["historical"] for row in tagged),
        "hotspots": {
            name: {
                "authors": len(spot["authors"]),
                "event": len(spot["event"]),
                "event_last_48h": len(spot["event_last_48h"]),
                "control_change": len(spot["control_change"]),
                "escalation": len(spot["escalation"]),
                "deescalation": len(spot["deescalation"]),
                "actors": sorted(spot["actors"]),
                "evidence": _evidence([by_id[pid] for pid in spot["posts"]]),
            }
            for name, spot in stats.items()
        },
    }


def _evidence(rows: list, limit: int = 3) -> list:
    """Most informative posts first: control changes, then events, then escalation."""
    def weight(row):
        return (row["control_change"], row["event"], row["escalation"], row["created_at"] or "")
    picked, prints = [], set()
    for row in sorted(rows, key=weight, reverse=True):
        if row["fingerprint"] in prints:
            continue
        prints.add(row["fingerprint"])
        picked.append({"id": row["id"], "created_at": row["created_at"], "text": row["text"][:280]})
        if len(picked) == limit:
            break
    return picked


def signal_multiplier(spot: dict) -> float:
    """How much this week's posts raise (or lower) the near-term hazard."""
    log_m = (
        0.25 * math.log1p(spot["event"])
        + 0.15 * math.log1p(spot["event_last_48h"])
        + 0.35 * math.log1p(spot["control_change"])
        + 0.20 * math.log1p(spot["escalation"])
        - 0.30 * math.log1p(spot["deescalation"])
    )
    return min(max(math.exp(log_m), 0.5), 6.0)


def probability(monthly_hazard: float, multiplier: float, days: int, decay: float = SIGNAL_DECAY_DAYS) -> float:
    """P(at least one major escalation within `days`).

    Hazard over time is base * (1 + (m - 1) * exp(-t / decay)), integrated from 0 to `days`.
    """
    expected = monthly_hazard / 30 * (days + (multiplier - 1) * decay * (1 - math.exp(-days / decay)))
    return 1 - math.exp(-max(expected, 0.0))


def forecast(posts: list, base_hazard: dict | None = None, tagged: list | None = None) -> dict:
    """Forecast from raw posts (rule tagging) or from rows already tagged by an extractor."""
    signals = hotspot_signals(tagged if tagged is not None else [classify(post) for post in posts])
    base_hazard = base_hazard or {}
    rows = []
    for name, (province, tier, _) in HOTSPOTS.items():
        spot = signals["hotspots"][name]
        base = base_hazard.get(name, TIER_HAZARD[tier])
        mult = signal_multiplier(spot)
        row = {
            "hotspot": name,
            "province": province,
            "tier": tier,
            "base_monthly_hazard": round(base, 3),
            "signal_multiplier": round(mult, 2),
            **{f"p{h}": round(probability(base, mult, h), 3) for h in HORIZONS},
            **{f"p{h}_baseline": round(probability(base, 1.0, h), 3) for h in HORIZONS},
            **{key: spot[key] for key in ("authors", "event", "event_last_48h", "control_change", "escalation", "deescalation", "actors", "evidence")},
        }
        row["p30_lift"] = round(row["p30"] - row["p30_baseline"], 3)
        rows.append(row)
    rows.sort(key=lambda row: (row["p30"], row["p30_lift"]), reverse=True)
    return {
        "usable_posts": signals["usable"],
        "dropped_off_topic": signals["dropped_off_topic"],
        "dropped_historical": signals["dropped_historical"],
        "rows": rows,
    }


def save_forecast(result: dict, summary: dict, path: Path | None = None, **meta) -> Path:
    """Write the forecast with resolve-by dates so it can be scored later."""
    issued = datetime.fromisoformat(max(summary["posts_per_day"]) + "T00:00:00+00:00") + timedelta(days=1)
    payload = {
        **meta,
        "issued": issued.date().isoformat(),
        "source_posts": summary["posts_file"],
        "target": "major escalation: >=10 deaths in one event, a locality changing hands, or 5,000+ newly displaced",
        "resolve_by": {f"p{h}": (issued + timedelta(days=h)).date().isoformat() for h in HORIZONS},
        "rows": [{k: v for k, v in row.items() if k != "evidence"} for row in result["rows"]],
    }
    if path is None:
        FORECASTS.mkdir(parents=True, exist_ok=True)
        path = FORECASTS / f"forecast_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def base_rates_from_acled(csv_path: Path, months: float) -> dict:
    """Monthly major-escalation hazard per hotspot from an ACLED export.

    Counts events with 10+ fatalities or a territory change, matched to hotspots
    by admin2/location. Pass the result to `forecast(posts, base_hazard=...)`.
    """
    counts = Counter()
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        for event in csv.DictReader(handle):
            major = int(float(event.get("fatalities") or 0)) >= 10 or "territory" in (event.get("sub_event_type") or "").lower()
            if not major:
                continue
            words = _words(normalize(f"{event.get('admin2', '')} {event.get('admin3', '')} {event.get('location', '')}"))
            for name, (_, _, aliases) in HOTSPOTS.items():
                if any(alias in words for alias in aliases):
                    counts[name] += 1
                    break
    return {name: counts[name] / months for name in HOTSPOTS if counts[name]}


def brier(forecast_path: Path, outcomes: dict) -> dict:
    """Score a saved forecast against the baseline-only forecast it was built on.

    `outcomes` maps hotspot -> days from the issue date to the first major escalation (None if none).
    Skill above 0 means the post signal beat the baseline.
    """
    saved = json.loads(Path(forecast_path).read_text())
    scores = {}
    for h in HORIZONS:
        model, base = [], []
        for row in saved["rows"]:
            if row["hotspot"] not in outcomes:
                continue
            days = outcomes[row["hotspot"]]
            happened = 1.0 if days is not None and days <= h else 0.0
            model.append((row[f"p{h}"] - happened) ** 2)
            base.append((row[f"p{h}_baseline"] - happened) ** 2)
        if not model:
            scores[f"p{h}"] = None
            continue
        model_score, base_score = sum(model) / len(model), sum(base) / len(base)
        scores[f"p{h}"] = {
            "n": len(model),
            "brier": round(model_score, 4),
            "brier_baseline": round(base_score, 4),
            "skill": round(1 - model_score / base_score, 3) if base_score else None,
        }
    return scores
