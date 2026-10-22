"""LLM event extraction for posts, returning the same tagged rows as `forecast.classify`.

Each post goes through one structured-output call, to Claude or to Gemini depending on
the model id ("gemini-..." goes to Gemini). Results are cached on disk by (post id, model,
prompt version), so re-running the pipeline or the evaluation does not pay for the same
post twice, and changing the prompt starts a fresh cache.
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import anthropic
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel, Field

import forecast as fc
from fetch_recent import load_env

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "llm_cache"
DEFAULT_MODEL = "claude-opus-5-5"
PROMPT_VERSION = "v1"
# Models that accept server-side refusal fallbacks (fallbacks="default").
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"}

Hotspot = Literal[tuple(fc.HOTSPOTS)]
Actor = Literal[tuple(fc.ACTORS)]


class PostExtraction(BaseModel):
    relevant: bool = Field(description="About armed conflict, insecurity or political violence in the DRC.")
    historical: bool = Field(description="The main event described happened before 2026, or the post is an anniversary or history thread.")
    event_types: list[Literal["armed_clash", "attack_on_civilians", "control_change", "displacement", "abduction", "shelling_or_drone"]] = Field(
        description="Violent events the post reports as happening now or in the last few weeks. Empty for opinion, rhetoric or history."
    )
    escalation: bool = Field(description="Signals of coming escalation: offensive being prepared, reinforcements, mobilisation, arms distribution, new alliances, ultimatums, encirclement.")
    deescalation: bool = Field(description="Signals of de-escalation: ceasefire, truce, peace talks or agreement, withdrawal, disarmament.")
    hotspots: list[Hotspot] = Field(description="Hotspots where the reported events or signals take place. Only places in the list; leave empty if none apply.")
    other_places: list[str] = Field(description="DRC place names in the post that do not fit any listed hotspot, as written.")
    actors: list[Actor] = Field(description="Armed or security actors involved.")
    fatalities: int | None = Field(description="Deaths reported for the current event, if a number is given.")
    displaced: int | None = Field(description="People reported newly displaced, if a number is given. Convert households to people only if the post does.")
    source_type: Literal["field_report", "news", "party_claim", "opinion", "other"] = Field(
        description="party_claim = a belligerent or its supporters claiming success or blaming the other side."
    )


SYSTEM = f"""You extract conflict events from posts on X about the Democratic Republic of the Congo.
Posts are in French, English, Swahili, Kinyarwanda or Lingala, and many are partisan.

Rules:
- Tag what the post reports, not what the author hopes or fears. Accusations and slogans are not events.
- A post about something that happened years ago (e.g. 1996 refugee camp attacks, the 2025 fall of Goma) is historical and gets no event_types.
- "Kinshasa" used to mean the government is not a location. Only tag the Kinshasa hotspot for events in the city itself.
- Map localities to their hotspot. Examples: Kaziba, Muhungu, Kamanyola and the Itara-Luvungi groupement are in Walungu; Luvungi village, Katogota, Sange and Kiliba are in the Uvira plain; Minembwe is in Fizi; Sake and Kitchanga are in Masisi; Pimbo and Fataki are in Djugu; Bunia is in Irumu.
- FRN is the coalition that includes M23/AFC. Tag M23/AFC and FRN separately as written.

Hotspots: {", ".join(fc.HOTSPOTS)}."""


def is_gemini(model: str) -> bool:
    return model.startswith("gemini")


def _client(model: str):
    env = load_env()
    if is_gemini(model):
        # A 60s cap per call; retries are handled in _call_gemini so they are visible and bounded.
        options = genai_types.HttpOptions(timeout=60_000, retry_options=genai_types.HttpRetryOptions(attempts=1))
        key = env.get("GEMINI_API_KEY")
        return genai.Client(api_key=key, http_options=options) if key else genai.Client(http_options=options)
    key = env.get("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()


# Posts about killings and sexual violence are the signal here; default filters would drop them.
GEMINI_SAFETY = [
    genai_types.SafetySetting(category=category, threshold="BLOCK_NONE")
    for category in ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                     "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")
]


def _user_message(post: dict) -> str:
    return f"Posted {post.get('created_at', '')[:10]}:\n\n{post.get('text', '')}"


def _call_gemini(client, post: dict, model: str, attempts: int = 5) -> dict:
    """Returns {"extraction", "error", "usage", "model"}; retries overload and rate-limit errors."""
    config = genai_types.GenerateContentConfig(
        system_instruction=SYSTEM,
        response_mime_type="application/json",
        response_schema=PostExtraction,
        safety_settings=GEMINI_SAFETY,
    )
    for attempt in range(attempts):
        try:
            response = client.models.generate_content(model=model, contents=_user_message(post), config=config)
            break
        except genai_errors.APIError as error:
            if error.code not in (429, 500, 503, 504) or attempt == attempts - 1:
                return {"extraction": None, "error": f"gemini {error.code}: {error.message}"[:300]}
            time.sleep(2 ** attempt * 3)
    meta = response.usage_metadata
    result = {
        "model": response.model_version or model,
        "usage": {
            "input": meta.prompt_token_count or 0,
            "output": (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0),
            "cache_read": meta.cached_content_token_count or 0,
        },
        "error": None,
        "extraction": None,
    }
    blocked = response.prompt_feedback and response.prompt_feedback.block_reason
    finish = response.candidates[0].finish_reason if response.candidates else None
    if blocked or (finish and finish.name in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST")):
        result["error"] = f"blocked: {blocked or finish.name}"
    elif response.parsed is None:
        result["error"] = f"no parsed output (finish_reason={finish.name if finish else None})"
    else:
        result["extraction"] = response.parsed.model_dump()
    return result


def _call_claude(client, post: dict, model: str) -> dict:
    kwargs = {}
    if model in FALLBACK_MODELS:
        kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    try:
        response = client.beta.messages.parse(
            model=model,
            max_tokens=2000,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": _user_message(post)}],
            output_config={"effort": "low"},
            output_format=PostExtraction,
            **kwargs,
        )
    except (anthropic.BadRequestError, anthropic.NotFoundError, anthropic.AuthenticationError):
        raise
    except anthropic.APIError as error:
        return {"extraction": None, "error": f"{type(error).__name__}: {error}"[:300]}
    result = {
        "model": response.model,
        "usage": {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
            "cache_read": response.usage.cache_read_input_tokens or 0,
        },
        "error": None,
        "extraction": None,
    }
    if response.stop_reason == "refusal":
        result["error"] = "refusal"
    elif response.parsed_output is None:
        result["error"] = f"no parsed output (stop_reason={response.stop_reason})"
    else:
        result["extraction"] = response.parsed_output.model_dump()
    return result


def _cache_path(post_id: str, model: str) -> Path:
    return CACHE / model / PROMPT_VERSION / f"{post_id}.json"


def extract_one(client, post: dict, model: str) -> dict:
    """One post -> {"extraction": dict | None, "usage": ..., "error": ...}. Cached."""
    path = _cache_path(post["id"], model)
    if path.exists():
        return json.loads(path.read_text())
    started = time.time()
    call = _call_gemini if is_gemini(model) else _call_claude
    record = {"prompt_version": PROMPT_VERSION, **call(client, post, model), "seconds": round(time.time() - started, 2)}
    if record["extraction"] is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))  # only successes are cached; failures retry next run
    return record


def extract(posts: list, model: str = DEFAULT_MODEL, workers: int = 4) -> list:
    """Extraction records for `posts`, in order."""
    client = _client(model)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda post: extract_one(client, post, model), posts))


def to_tagged(post: dict, record: dict) -> dict:
    """Convert an extraction record to the row shape `forecast.classify` produces."""
    row = fc.classify(post)  # keeps id, author, text and fingerprint identical across extractors
    found = record.get("extraction")
    if found is None:
        row.update(on_topic=False, historical=False, hotspots=[], actors=[], extraction_error=record.get("error"))
        for signal in fc.SIGNALS:
            row[signal] = False
        return row
    types = set(found["event_types"])
    row.update(
        on_topic=found["relevant"],
        historical=found["historical"],
        hotspots=found["hotspots"],
        actors=found["actors"],
        event=bool(types),
        control_change="control_change" in types,
        escalation=found["escalation"],
        deescalation=found["deescalation"],
        fatalities=found["fatalities"],
        displaced=found["displaced"],
        other_places=found["other_places"],
        source_type=found["source_type"],
    )
    return row


def tag_posts(posts: list, model: str = DEFAULT_MODEL) -> tuple[list, dict]:
    """Tagged rows plus a cost/error summary for the run manifest."""
    records = extract(posts, model)
    usage = {"input": 0, "output": 0, "cache_read": 0}
    errors = [record["error"] for record in records if record.get("error")]
    for record in records:
        for key in usage:
            usage[key] += (record.get("usage") or {}).get(key, 0)
    stats = {
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "posts": len(posts),
        "errors": len(errors),
        "error_examples": sorted(set(re.sub(r"\d+", "#", error) for error in errors))[:5],
        "tokens": usage,
    }
    return [to_tagged(post, record) for post, record in zip(posts, records)], stats
