"""Download ACLED events for the DRC into data/acled/, for base rates and backtests.

  python acled.py --months 12

ACLED has no API key any more: scripts log in with the myACLED email and password
(OAuth password grant) and get a 24-hour token. Put ACLED_EMAIL and ACLED_PASSWORD
in .env. Which rows you get depends on your myACLED access level.
"""

import argparse
import csv
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

from fetch_recent import ROOT, load_env

TOKEN_URL = "https://acleddata.com/oauth/token"
READ_URL = "https://acleddata.com/api/acled/read"
OUT = ROOT / "data" / "acled"
PAGE_SIZE = 5000


def _credential(name: str) -> str:
    value = load_env().get(name) or os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} is missing from .env and the environment")
    return value


def token() -> str:
    body = urllib.parse.urlencode({
        "username": _credential("ACLED_EMAIL"),
        "password": _credential("ACLED_PASSWORD"),
        "grant_type": "password",
        "client_id": "acled",
        "scope": "authenticated",
    }).encode()
    request = urllib.request.Request(TOKEN_URL, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)["access_token"]
    except urllib.error.HTTPError as error:
        body = error.read()
        if b"_cf_chl" in body or b"Just a moment" in body:
            raise RuntimeError(
                "ACLED's Cloudflare protection blocked the login from this network (a browser challenge, "
                "not a credentials error). Download a CSV from the myACLED Data Export Tool instead and "
                "pass it with --acled, or ask access@acleddata.com to allow API access."
            ) from None
        raise RuntimeError(f"ACLED login failed (HTTP {error.code}): {body[:300]!r}") from None


def fetch(access_token: str, country: str, start: date, end: date) -> list:
    rows, page = [], 1
    while True:
        query = urllib.parse.urlencode({
            "_format": "json",
            "country": country,
            "event_date": f"{start.isoformat()}|{end.isoformat()}",
            "event_date_where": "BETWEEN",
            "limit": PAGE_SIZE,
            "page": page,
        })
        request = urllib.request.Request(f"{READ_URL}?{query}", headers={"Authorization": f"Bearer {access_token}"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"ACLED read failed (HTTP {error.code}): {error.read()[:300]!r}") from None
        batch = payload.get("data") or []
        rows += batch
        print(f"page {page}: {len(batch)} events")
        if len(batch) < PAGE_SIZE:
            return rows
        page += 1


def save(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--country", default="Democratic Republic of Congo")
    args = parser.parse_args()

    end = date.today()
    start = end - timedelta(days=round(args.months * 30.44))
    rows = fetch(token(), args.country, start, end)
    path = OUT / f"acled_drc_{start.isoformat()}_{end.isoformat()}.csv"
    save(rows, path)
    print(f"{len(rows)} events -> {path}")
    if not rows:
        print("No rows: your myACLED access level may not include event-level data for this period.")
    else:
        print(f"use with: python pipeline.py run --acled {path} --acled-months {args.months}")


if __name__ == "__main__":
    main()
