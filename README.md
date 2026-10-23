# DRC escalation watch from X posts

Forecasts, for 20 DRC hotspots, the probability of a **major escalation** within 30, 60 and 90 days: a single event with 10+ deaths, a locality changing hands, or 5,000+ newly displaced. "Any violence" is not the target because it is close to certain every month in most eastern territories.

**Live dashboard: [mhadeli.github.io/drc-escalation-watch](https://mhadeli.github.io/drc-escalation-watch/)**. It is built only from the saved runs in `history/runs`, with no API calls, and shows forecasts by hotspot, a map, trends, scores and run health. No post text is published.

## Pipeline

```
ingest ─▶ extract ─▶ forecast ─▶ health ─▶ brief          score (once resolve-by dates pass)
X API     rules or    base hazard  checks     one page      Brier vs baseline-only forecast
          LLM         × signal
```

Keys go in `.env` (gitignored): `X_BEARER_TOKEN`, plus `GEMINI_API_KEY` or `ANTHROPIC_API_KEY` for LLM extraction, and `ACLED_EMAIL` / `ACLED_PASSWORD` for base rates.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python pipeline.py run                                   # latest saved pull, rule extractor
.venv/bin/python pipeline.py run --fetch --extractor gemini-3.8-flash
.venv/bin/python acled.py --months 12                              # ACLED events -> data/acled/
.venv/bin/python pipeline.py run --acled data/acled/<file>.csv
.venv/bin/python pipeline.py publish                                # latest run -> history/runs
.venv/bin/python pipeline.py score                                  # against history/outcomes.csv
.venv/bin/python -m unittest discover -s tests
```

Each run writes `data/runs/<run_id>/`:

| File | What it holds |
|---|---|
| `manifest.json` | Inputs, code hash, extractor and prompt version, token use, counts, timings, warnings |
| `tagged.jsonl` | Every post with hotspots, actors and signal tags |
| `forecast.json` | Probabilities, the baseline-only probabilities, and resolve-by dates |
| `health.json` | Usable share, extraction errors, change since last run, places missing from the gazetteer |
| `brief.md` | One page for readers who don't build models |

### Saved data

| Where | What | In git? |
|---|---|---|
| `data/raw/`, `data/runs/<id>/tagged.jsonl`, `brief.md` | Posts, extractions, briefs with evidence | No: local only (X terms) |
| `data/llm_cache/` | Gemini/Claude responses per post | No: local only |
| `history/runs/<id>/` | `manifest.json`, `forecast.json`, `health.json`, `scores.json` | Yes |
| `history/outcomes.csv` | Major escalations you record, for scoring | Yes |

Everything downstream of extraction runs from saved files. Once the APIs are retired, `pipeline.py run` (no `--fetch`) on a saved pull and `pipeline.py score` still work, and LLM extractions are reused from `data/llm_cache/`.

### GitHub Actions

- `tests.yml` runs the unit tests on every push.
- `dashboard.yml` rebuilds the dashboard when `history/` changes. It reads saved files only.
- `weekly.yml` is manual only (**Actions → weekly-forecast → Run workflow**). It pulls from X, extracts with `gemini-3.8-flash`, then publishes the run to `history/runs` on `main`. It needs repository secrets `X_BEARER_TOKEN` and `GEMINI_API_KEY`. Nothing runs on a schedule, so no credits are spent unless you start it.

Publishing locally works the same way: `pipeline.py run --fetch --extractor gemini-3.8-flash`, then `pipeline.py publish`, then commit `history/`.

### Dashboard (GitHub Pages)

The page has five sections:
- forecast per hotspot against its baseline, for 30/60/90 days
- a map
- forecasts over time
- scores against outcomes
- run health

One-time setup: **Settings → Pages → Source: GitHub Actions**. To preview locally, run `python dashboard.py --runs history/runs` and open `site/index.html`.

## Modules

- `fetch_recent.py`: X recent search (last 7 days only).
- `forecast.py`: gazetteer, rule tagging, hazard model, ACLED base rates, Brier scoring.
- `extract_llm.py`: LLM structured-output extraction (Gemini for `gemini-*` model ids, Claude otherwise) into the same row shape as the rules, cached by post, model and prompt version.
- `acled.py`: ACLED download via myACLED OAuth login (email and password; ACLED no longer issues API keys).
- `evaluate.py`: labelling sheet and field-level metrics for any extractor.
- `pipeline.py`: orchestration, health checks, brief, scoring.
- `dashboard.py` + `dashboard/index.html`: static dashboard built from saved runs.

## Model

Monthly hazard per hotspot (priors in `TIER_HAZARD`, or ACLED rates via `--acled`) times a signal multiplier from unique-author counts of event, control-change, escalation and de-escalation posts. The boost decays with a 21-day time constant, so the posts move the 30-day number most:

P(h) = 1 − exp(−λ/30 · (h + (m − 1) · τ · (1 − e^(−h/τ))))

## Evaluation

**Extraction.** Run `python evaluate.py sample`, label `data/labels/to_label.csv` without looking at any extractor's output, save it as `data/labels/gold.csv`, then:

```bash
.venv/bin/python evaluate.py score --model gemini-3.8-flash --model claude-opus-5-5
```

This reports precision, recall and F1 per field, splits by language, lists errors, and gives token use per model. Rules are always included as the baseline.

**Forecasts.** Record each major escalation in `history/outcomes.csv` (`hotspot,date,source,note`; a hotspot can be a short name such as `Kaziba`). `pipeline.py score` reports Brier score for the model and for the baseline-only forecast, plus skill = 1 − Brier/Brier_baseline. Skill above 0 means the X signal adds information.

## Known limits

- X volume tracks attention as much as violence, and partisan accounts on every side post victory claims.
- Baseline hazards are priors until replaced with ACLED rates; the probabilities are uncalibrated until scored.
- Gemini's free tier runs out within a few posts; weekly runs of 1,000 posts need billing enabled.
- The rules can't tell the past from the present inside a post ("Goma était tombée…"). That is the main case the LLM extractor has to beat.

## Author

Hussain Adeli ([@mhadeli](https://github.com/mhadeli))
