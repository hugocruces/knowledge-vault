#!/usr/bin/env python3
"""
review.py — End-to-end literature review generator from your Zotero library.

Usage:
    review.py "your research question" [options]

Options:
    --limit N        Papers to include in synthesis (default: 12)
    --candidates N   Candidate papers fetched before re-ranking (default: 40)
    --out FILE       Output markdown file (default: auto-named from query)

Pipeline:
    1. Full-text search Zotero for candidate papers
    2. Claude re-ranks and selects the most relevant subset
    3. Full texts fetched via Zotero API and cached in text_cache/{key}.txt
    4. Claude synthesises a structured literature review → markdown
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path

import fitz          # PyMuPDF — fallback PDF extraction
import anthropic
from dotenv import load_dotenv
from pyzotero import zotero

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID", "")
ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY", "")
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL   = "claude-sonnet-4-6"

TEXT_CACHE         = HERE / "text_cache"
MAX_CHARS_PER_PAPER = 40_000   # ~10k tokens; enough for detailed synthesis
ZOTERO_API_BASE    = "https://api.zotero.org"


# ── Zotero helpers ─────────────────────────────────────────────────────────────

def search_zotero(zot, query: str, limit: int) -> list[dict]:
    """Full-text search across the entire Zotero library."""
    return zot.items(q=query, qmode="everything", limit=limit, itemType="-attachment")


def fetch_fulltext_api(item_key: str) -> str:
    """
    Pull full text from the Zotero API fulltext endpoint.
    Returns empty string if not indexed or on error.
    """
    url = f"{ZOTERO_API_BASE}/users/{ZOTERO_USER_ID}/items/{item_key}/fulltext"
    try:
        r = requests.get(
            url,
            headers={"Zotero-API-Key": ZOTERO_API_KEY},
            timeout=15,
        )
        if r.status_code == 200:
            return r.json().get("content", "")
        return ""
    except Exception as e:
        print(f"    [fulltext API] {e}")
        return ""


def fetch_fulltext_pdf(zot, item_key: str) -> str:
    """
    Fallback: find the PDF in ~/Zotero/storage and extract with PyMuPDF.
    """
    storage = Path.home() / "Zotero" / "storage"
    try:
        children = zot.children(item_key)
    except Exception:
        return ""
    for child in children:
        if child["data"].get("contentType") == "application/pdf":
            att_key = child["key"]
            filename = child["data"].get("filename", "")
            pdf_path = storage / att_key / filename
            if pdf_path.exists():
                try:
                    doc = fitz.open(str(pdf_path))
                    return "\n\n".join(page.get_text() for page in doc)
                except Exception as e:
                    print(f"    [PyMuPDF] {e}")
    return ""


def get_full_text(zot, item_key: str) -> str:
    """
    Return full text for item_key, using cache if available.
    Tries Zotero API first, falls back to local PDF extraction.
    """
    cache_path = TEXT_CACHE / f"{item_key}.txt"
    if cache_path.exists():
        print(f"    [cache hit] {item_key}.txt")
        return cache_path.read_text(encoding="utf-8")

    print(f"    Fetching via API…")
    text = fetch_fulltext_api(item_key)

    if not text.strip():
        print(f"    API empty — trying local PDF…")
        text = fetch_fulltext_pdf(zot, item_key)

    text = text[:MAX_CHARS_PER_PAPER]

    if text.strip():
        TEXT_CACHE.mkdir(exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
        print(f"    Cached → {cache_path.name}")
    else:
        print(f"    WARNING: no text found for {item_key}")

    return text


def get_item_metadata(zot, item_key: str) -> dict:
    item = zot.item(item_key)
    data = item["data"]
    authors = ", ".join(
        f"{c.get('lastName', '')} {c.get('firstName', '')[:1]}."
        for c in data.get("creators", [])
        if c.get("creatorType") == "author"
    )
    return {
        "key":     item_key,
        "title":   data.get("title", ""),
        "authors": authors or "(unknown)",
        "year":    (data.get("date") or "")[:4],
        "doi":     data.get("DOI", ""),
    }


# ── Claude helpers ─────────────────────────────────────────────────────────────

def rerank_with_claude(client, query: str, items: list[dict], top_n: int) -> list[str]:
    """Ask Claude to pick the top_n most relevant item keys for the query."""
    candidates = []
    for item in items:
        data = item["data"]
        title = data.get("title", "(no title)")
        authors = ", ".join(
            f"{c.get('lastName', '')}"
            for c in data.get("creators", [])
            if c.get("creatorType") == "author"
        )
        year = (data.get("date") or "")[:4]
        abstract = (data.get("abstractNote") or "")[:400]
        candidates.append(
            f"Key: {item['key']}\nTitle: {title}\nAuthors: {authors} ({year})\nAbstract: {abstract}"
        )

    prompt = (
        f'Research question: "{query}"\n\n'
        f"Select the {top_n} most relevant papers from the list below. "
        f"Return ONLY a JSON array of item keys (strings), ordered from most to least relevant. "
        f"No explanation, no markdown fences.\n\n"
        + "\n---\n".join(candidates)
    )

    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        keys = json.loads(raw)
        return [k for k in keys if isinstance(k, str)]
    except json.JSONDecodeError:
        print(f"  WARNING: Claude returned invalid JSON for re-ranking:\n{raw[:300]}")
        # Fall back to first top_n items
        return [item["key"] for item in items[:top_n]]


def synthesize(client, query: str, papers: list[dict], out_path: Path) -> None:
    """Synthesise all full texts into a structured literature review."""
    ref_list = "\n".join(
        f"- [{p['key']}] {p['authors']} ({p['year']}). {p['title']}."
        for p in papers
    )

    corpus_parts = []
    for p in papers:
        text = p["text"] or "(full text unavailable — use abstract only)"
        corpus_parts.append(
            f"=== [{p['key']}] {p['authors']} ({p['year']}) — {p['title']} ===\n{text}"
        )
    full_corpus = "\n\n".join(corpus_parts)

    total_chars = len(full_corpus)
    print(f"  Corpus: {len(papers)} papers, {total_chars:,} chars (~{total_chars // 4:,} tokens)")

    prompt = f"""You are a senior research economist writing a structured literature review.

Research question: "{query}"

Below are the full texts (or excerpts) of {len(papers)} papers. Synthesise them into a ~2-page literature review in markdown following these rules:

- Title: "# [topic]: Evidence and Implications"
- Sections: Introduction, then 3–4 thematic sections with descriptive headings, Policy Implications, Conclusion
- Written in flowing paragraphs — no bullet points in the body
- Each paragraph must be a single continuous line (no mid-paragraph line breaks)
- Cite inline as (Author, Year) — only papers listed below
- End with a References table: | # | Authors | Year | Title |
- Do not invent references or cite papers not in the list

Available papers:
{ref_list}

--- FULL TEXTS ---
{full_corpus}
"""

    print(f"  Sending to Claude ({len(prompt):,} chars)…")
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    result = msg.content[0].text.strip()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result, encoding="utf-8")
    print(f"\nReview written to: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a literature review from your Zotero library."
    )
    parser.add_argument("query", help="Research question or topic")
    parser.add_argument(
        "--limit", type=int, default=12,
        help="Number of papers to include in the synthesis (default: 12)",
    )
    parser.add_argument(
        "--candidates", type=int, default=40,
        help="Candidate papers to retrieve before re-ranking (default: 40)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output markdown file (default: auto-named from query)",
    )
    args = parser.parse_args()

    missing = [k for k, v in {
        "ANTHROPIC_API_KEY": ANTHROPIC_KEY,
        "ZOTERO_API_KEY":    ZOTERO_API_KEY,
        "ZOTERO_USER_ID":    ZOTERO_USER_ID,
    }.items() if not v]
    if missing:
        print(f"ERROR: missing config: {', '.join(missing)}")
        sys.exit(1)

    if args.out:
        out_path = Path(args.out)
    else:
        slug = args.query[:50].strip().replace(" ", "_").replace("/", "-")
        out_path = HERE / f"{slug}.md"

    zot    = zotero.Zotero(ZOTERO_USER_ID, "user", ZOTERO_API_KEY)
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    # ── Step 1: Search ──────────────────────────────────────────────────────────
    print(f"\n[1/4] Searching Zotero: {args.query!r} (up to {args.candidates} candidates)…")
    candidates = search_zotero(zot, args.query, limit=args.candidates)
    print(f"      Found {len(candidates)} candidates.")
    if not candidates:
        print("No results. Try a broader query.")
        sys.exit(1)

    # ── Step 2: Re-rank ─────────────────────────────────────────────────────────
    top_n = min(args.limit, len(candidates))
    print(f"\n[2/4] Re-ranking with Claude → selecting top {top_n}…")
    selected_keys = rerank_with_claude(client, args.query, candidates, top_n)
    print(f"      Selected keys: {selected_keys}")

    # ── Step 3: Fetch full texts ────────────────────────────────────────────────
    print(f"\n[3/4] Fetching full texts (cache: {TEXT_CACHE})…")
    papers = []
    for key in selected_keys:
        meta = get_item_metadata(zot, key)
        print(f"  {meta['authors']} ({meta['year']}) — {meta['title'][:65]}")
        text = get_full_text(zot, key)
        papers.append({**meta, "text": text})

    # ── Step 4: Synthesise ──────────────────────────────────────────────────────
    print(f"\n[4/4] Synthesising literature review…")
    synthesize(client, args.query, papers, out_path)


if __name__ == "__main__":
    main()
