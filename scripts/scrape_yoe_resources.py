#!/usr/bin/env python3
"""
scrape_yoe_resources.py — scrape Soil Werks + Wele Waters into episode-shaped
JSON files for ingestion into Pinecone.

Sources (per Aaron's Apr 28 email):
  - https://yonearth.org/soilwerks/                (Soil Werks Biodynamic Fertilizer)
  - https://yonearth.org/wele-waters/              (Wele Waters social-enterprise page)
  - https://welewaters.com/                        (Wele Waters product site)

Strategy: yonearth.org pages are fetched via the WP REST API
(`/wp-json/wp/v2/pages?slug=<slug>`) which returns clean structured JSON.
welewaters.com is fetched as plain HTML and parsed with BeautifulSoup.

Output:
  data/transcripts/episode_resource_soilwerks.json
  data/transcripts/episode_resource_welewaters.json   (combined yonearth.org page +
                                                       welewaters.com homepage)

Episode-shaped JSON keys (required by src/ingestion/episode_processor.py):
  episode_number, title, url, guest_name, full_transcript,
  source_type, source_url   (the latter two are ignored by Episode class but
                             survive through to chunk metadata for filtering)

Gardening Course (academy.yonearth.org/garden-success-bundle/) is **deferred** —
content is gated behind ThriveCart purchase even with code GRATIS, and
extracting course-content pages requires browser automation through the checkout
flow. Tracked as follow-up.

Usage:
  python3 scripts/scrape_yoe_resources.py
"""
from __future__ import annotations

import html
import json
import pathlib
import re
import sys
import time
import urllib.request
import urllib.error

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "transcripts"

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def fetch(url: str, accept_json: bool = False) -> str:
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "application/json" if accept_json else "text/html,*/*",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def strip_html(s: str) -> str:
    """Lightweight HTML→text. No bs4 dep — uses regex + html.unescape."""
    if not s:
        return ""
    # Drop script/style blocks entirely
    s = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", s, flags=re.S | re.I)
    # Normalize <br> and <p> to line breaks before tag strip
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</p>", "\n\n", s, flags=re.I)
    # Strip remaining tags
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    # Collapse whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def fetch_yonearth_page(slug: str) -> dict | None:
    """Fetch a yonearth.org page via WP REST API; return rendered title + content."""
    url = f"https://yonearth.org/wp-json/wp/v2/pages?slug={slug}"
    raw = fetch(url, accept_json=True)
    arr = json.loads(raw)
    if not arr:
        return None
    p = arr[0]
    return {
        "url": p.get("link") or f"https://yonearth.org/{slug}/",
        "title": strip_html(p.get("title", {}).get("rendered", "") or "").strip(),
        "content_html": p.get("content", {}).get("rendered", "") or "",
        "modified": p.get("modified", ""),
    }


def fetch_welewaters_homepage() -> str:
    """Plain HTML scrape of welewaters.com — strip tags, return body text."""
    raw = fetch("https://welewaters.com/")
    # Try to extract just the <body>… section to avoid head/script noise
    m = re.search(r"<body[^>]*>(.*?)</body>", raw, flags=re.S | re.I)
    body = m.group(1) if m else raw
    return strip_html(body)


def write_episode_json(filename: str, payload: dict) -> pathlib.Path:
    out = OUT_DIR / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def build_soilwerks() -> dict:
    page = fetch_yonearth_page("soilwerks")
    if not page:
        raise RuntimeError("yonearth.org/soilwerks/ returned no page")
    transcript = strip_html(page["content_html"])
    return {
        "episode_number": "resource_soilwerks",
        "title": page["title"] or "Soil Werks Biodynamic Fertilizer",
        "guest_name": "YOE Soil Werks",
        "url": page["url"],
        "publish_date": page["modified"][:10] if page["modified"] else "",
        "subtitle": "YOE social enterprise — biodynamic fertilizer & soil resources",
        "full_transcript": transcript,
        "source_type": "product",
        "source_url": page["url"],
        "is_yoe_enterprise": True,
    }


def build_welewaters() -> dict:
    yp = fetch_yonearth_page("wele-waters")
    if not yp:
        raise RuntimeError("yonearth.org/wele-waters/ returned no page")
    yp_text = strip_html(yp["content_html"])
    # Be polite to the second site
    time.sleep(0.5)
    ww_text = fetch_welewaters_homepage()
    combined = (
        f"=== From yonearth.org/wele-waters/ ===\n\n{yp_text}\n\n"
        f"=== From welewaters.com (homepage) ===\n\n{ww_text}\n"
    )
    return {
        "episode_number": "resource_welewaters",
        "title": yp["title"] or "Wele Waters — Biodynamic Hemp-Infused Aromatherapy Soaking Salts",
        "guest_name": "YOE Wele Waters",
        "url": yp["url"],
        "publish_date": yp["modified"][:10] if yp["modified"] else "",
        "subtitle": "YOE social enterprise — biodynamic hemp-infused soaking salts (welewaters.com)",
        "full_transcript": combined,
        "source_type": "external_partner",
        "source_url": "https://welewaters.com/",
        "is_yoe_enterprise": True,
    }


def main() -> int:
    print("→ scraping Soil Werks…", file=sys.stderr)
    sw = build_soilwerks()
    p_sw = write_episode_json("episode_resource_soilwerks.json", sw)
    print(f"  wrote {p_sw}  (transcript len: {len(sw['full_transcript'])})", file=sys.stderr)

    print("→ scraping Wele Waters…", file=sys.stderr)
    ww = build_welewaters()
    p_ww = write_episode_json("episode_resource_welewaters.json", ww)
    print(f"  wrote {p_ww}  (transcript len: {len(ww['full_transcript'])})", file=sys.stderr)

    print("\nDONE. Review the two JSON files before running the ingest script.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
