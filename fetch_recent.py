"""Pull recent public posts. The notebook calls these functions."""

import json
import os
import subprocess
import time
import urllib.parse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
TARGET = 1000
PAGE_SIZE = 100

QUERY = (
    '(Goma OR "North Kivu" OR "Nord-Kivu" OR "South Kivu" OR "Sud-Kivu" '
    'OR Ituri OR M23 OR "DR Congo" OR "RD Congo" OR RDC '
    'OR "Democratic Republic of the Congo" OR "République démocratique du Congo") '
    "(attaque OR attack OR combat OR violence OR affrontement OR war OR conflict "
    "OR displaced OR déplacés OR fighting OR bombardement OR offensive) "
    "-is:retweet"
)


def load_env(path: Path | None = None) -> dict:
    env_path = path or (ROOT / ".env")
    env = {}
    if not env_path.exists():
        return env
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def bearer_token() -> str:
    token = load_env().get("X_BEARER_TOKEN") or os.environ.get("X_BEARER_TOKEN", "")
    if not token:
        raise RuntimeError("X_BEARER_TOKEN is missing from .env and the environment")
    return token


class XApiError(RuntimeError):
    def __init__(self, status: str, body: str):
        super().__init__(f"X API HTTP {status}: {body[:800]}")
        self.status = status


# Out of credits or rate-limited: keep the posts already paid for instead of losing the run.
STOP_STATUSES = {"402", "429"}


def request_page(token: str, query: str, next_token: str | None, page_size: int) -> dict:
    params = {
        "query": query,
        "max_results": page_size,
        "tweet.fields": "id,text,created_at,lang,author_id,public_metrics,geo,entities,conversation_id",
        "expansions": "author_id,geo.place_id",
        "user.fields": "username,name,public_metrics,description,location",
    }
    if next_token:
        params["next_token"] = next_token
    url = "https://api.x.com/2/tweets/search/recent?" + urllib.parse.urlencode(params)
    result = subprocess.run(
        [
            "curl",
            "-sS",
            "-w",
            "\n%{http_code}",
            "-H",
            f"Authorization: Bearer {token}",
            url,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Request failed: {result.stderr.strip()[:400]}")
    body, _, status = result.stdout.rpartition("\n")
    if status != "200":
        raise XApiError(status, body)
    return json.loads(body)


def collect(token: str, query: str, target: int, page_size: int = PAGE_SIZE, pause_seconds: float = 1.0):
    """Page through recent search until `target` unique posts or the API runs out."""
    posts = {}
    users = {}
    places = {}
    next_token = None
    pages = 0
    stopped = None

    while len(posts) < target:
        pages += 1
        try:
            payload = request_page(token, query, next_token, page_size)
        except XApiError as error:
            if error.status not in STOP_STATUSES or not posts:
                raise
            stopped = str(error)[:200]
            print(f"page {pages}: stopped early, keeping {len(posts)} posts ({stopped})")
            break
        batch = payload.get("data") or []
        includes = payload.get("includes") or {}
        for user in includes.get("users") or []:
            users[user["id"]] = user
        for place in includes.get("places") or []:
            places[place["id"]] = place
        for post in batch:
            posts[post["id"]] = post
        meta = payload.get("meta") or {}
        print(
            f"page {pages}: {len(batch)} posts, {len(posts)} unique, "
            f"result_count={meta.get('result_count')}"
        )
        next_token = meta.get("next_token")
        if not next_token or not batch:
            break
        time.sleep(pause_seconds)

    ordered = list(posts.values())[:target]
    return {
        "posts": ordered,
        "users": users,
        "places": places,
        "pages": pages,
        "fetched": len(posts),
        "stopped": stopped,
    }


def save_run(collected: dict, query: str, requested: int) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    posts = collected["posts"]
    posts_path = RAW / f"drc_posts_{stamp}.jsonl"
    users_path = RAW / f"drc_users_{stamp}.jsonl"
    summary_path = RAW / f"drc_summary_{stamp}.json"

    write_jsonl(posts_path, posts)
    write_jsonl(users_path, collected["users"].values())

    languages = Counter(post.get("lang") or "und" for post in posts)
    days = Counter((post.get("created_at") or "")[:10] for post in posts)
    summary = {
        "query": query,
        "requested": requested,
        "returned": len(posts),
        "fetched_before_cap": collected["fetched"],
        "pages": collected["pages"],
        "unique_authors": len({post.get("author_id") for post in posts}),
        "languages": dict(languages.most_common()),
        "posts_per_day": dict(sorted(days.items())),
        "posts_file": str(posts_path),
        "users_file": str(users_path),
        "places": len(collected["places"]),
        "stopped_early": collected.get("stopped"),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def write_jsonl(path: Path, rows) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> list:
    text = path.read_text()
    decoder = json.JSONDecoder()
    rows = []
    index = 0
    size = len(text)
    while index < size:
        while index < size and text[index].isspace():
            index += 1
        if index >= size:
            break
        row, index = decoder.raw_decode(text, index)
        rows.append(row)
    return rows


def latest_summary() -> dict:
    files = sorted(RAW.glob("drc_summary_*.json"))
    if not files:
        raise FileNotFoundError(f"No saved runs in {RAW}")
    return json.loads(files[-1].read_text())


def main() -> None:
    collected = collect(bearer_token(), QUERY, TARGET, PAGE_SIZE)
    summary = save_run(collected, QUERY, TARGET)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
