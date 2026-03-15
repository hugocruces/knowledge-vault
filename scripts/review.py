#!/usr/bin/env python3
"""
review.py — End-to-end literature review generator from your Zotero library.

Usage:
    review.py "your research question" [options]

Options:
    --limit N        Papers to include in synthesis (default: 12)
    --candidates N   Candidate papers fetched before re-ranking (default: 40)
    --out FILE       Output markdown file (default: auto-named from query)
    --collection KEY Restrict search to a specific Zotero collection key

Pipeline:
    1. Full-text search Zotero for candidate papers
    2. Claude re-ranks and selects the most relevant subset
    3. Full texts extracted from local PDFs (pymupdf4llm → markdown)
       and cached in text_cache/{key}.md
    4. Claude synthesises a structured literature review → markdown

Text extraction uses pymupdf4llm which produces structured markdown
(headings, emphasis, tables) from PDFs — much better for LLM consumption
than raw text. Falls back to plain PyMuPDF if pymupdf4llm is unavailable.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pyzotero import zotero

HERE = Path(__file__).resolve().parent.parent   # repo root (scripts/ is one level down)
load_dotenv(HERE / ".env")

ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID", "")
ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY", "")
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL   = "claude-sonnet-4-6"

TEXT_CACHE     = HERE / "text_cache"
ZOTERO_STORAGE = Path.home() / "Zotero" / "storage"

# Context windows as of 2026-03 (update when models change):
#   claude-haiku-4-5   → 200k tokens
#   claude-sonnet-4-6  → 200k tokens
#   claude-opus-4-6    → 200k tokens
# Rule of thumb: ~4 chars per token; reserve ~20k tokens for prompt + output.
# Per-paper budget = (context_tokens - 20_000) * 4 / num_papers
# Approximate full-paper read capacity:
#   5  papers → ~144k chars/paper   (most papers fit fully)
#   10 papers →  72k chars/paper    (short papers fit; long papers trimmed)
#   15 papers →  48k chars/paper    (significant trimming likely)
CONTEXT_TOKENS = 200_000
PROMPT_RESERVE =  20_000   # tokens reserved for prompt scaffolding + output
CONTEXT_BUDGET = (CONTEXT_TOKENS - PROMPT_RESERVE) * 4  # chars available for corpus


# ── Text extraction ───────────────────────────────────────────────────────────

def _find_pdf_path(zot, item_key: str) -> Path | None:
    """Locate the PDF attachment for an item in ~/Zotero/storage/."""
    try:
        children = zot.children(item_key)
    except Exception:
        return None
    for child in children:
        if child["data"].get("contentType") == "application/pdf":
            att_key = child["key"]
            filename = child["data"].get("filename", "")
            pdf_path = ZOTERO_STORAGE / att_key / filename
            if pdf_path.exists():
                return pdf_path
    # Some items store the PDF directly under the item key
    item_dir = ZOTERO_STORAGE / item_key
    if item_dir.exists():
        for pdf in item_dir.glob("*.pdf"):
            return pdf
    return None


def _extract_markdown(pdf_path: Path) -> str:
    """
    Extract text from a PDF as structured markdown using pymupdf4llm.
    Falls back to plain PyMuPDF get_text() if pymupdf4llm is unavailable.
    """
    try:
        import pymupdf4llm
        return pymupdf4llm.to_markdown(str(pdf_path))
    except ImportError:
        pass
    except Exception as e:
        print(f"    [pymupdf4llm error: {e}] falling back to plain extraction")

    # Fallback: plain PyMuPDF
    import fitz
    doc = fitz.open(str(pdf_path))
    return "\n\n".join(page.get_text() for page in doc)


def get_full_text(zot, item_key: str, max_chars: int | None = None) -> str:
    """
    Return full text for item_key, using cache if available.
    Extracts from the local PDF in ~/Zotero/storage/ using pymupdf4llm.
    Full extracted text is always cached untruncated; max_chars is applied at read time.
    """
    # Check cache (.md only)
    cache_path = TEXT_CACHE / f"{item_key}.md"
    if cache_path.exists():
        print(f"    [cache hit] {cache_path.name}")
        text = cache_path.read_text(encoding="utf-8")
        return text[:max_chars] if max_chars else text

    # Find and extract from local PDF
    pdf_path = _find_pdf_path(zot, item_key)
    if not pdf_path:
        print(f"    WARNING: no PDF found for {item_key}")
        return ""

    print(f"    Extracting: {pdf_path.name[:60]}…")
    text = _extract_markdown(pdf_path)

    if text.strip():
        TEXT_CACHE.mkdir(exist_ok=True)
        out = TEXT_CACHE / f"{item_key}.md"
        out.write_text(text, encoding="utf-8")   # cache full text, untruncated
        print(f"    Cached → {out.name} ({len(text):,} chars)")
    else:
        print(f"    WARNING: no text extracted from {pdf_path.name}")

    return text[:max_chars] if max_chars else text


# ── Zotero helpers ─────────────────────────────────────────────────────────────

def search_zotero(zot, query: str, limit: int) -> list[dict]:
    """Full-text search across the entire Zotero library."""
    return zot.items(q=query, qmode="everything", limit=limit, itemType="-attachment")


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
        help=(
            "Number of papers to include in the synthesis (default: 12). "
            f"Context budget ({CONTEXT_TOKENS:,} token window, {CONTEXT_BUDGET:,} chars for corpus): "
            f"5 papers → ~{CONTEXT_BUDGET//5:,} chars/paper (full papers); "
            f"10 → ~{CONTEXT_BUDGET//10:,}; "
            f"15 → ~{CONTEXT_BUDGET//15:,} (trimming likely). "
            "Use fewer papers for deeper full-text coverage."
        ),
    )
    parser.add_argument(
        "--candidates", type=int, default=40,
        help="Candidate papers to retrieve before re-ranking (default: 40)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output markdown file (default: auto-named from query)",
    )
    parser.add_argument(
        "--collection", default=None,
        help="Restrict to a Zotero collection key (skips search, uses all items in collection)",
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
        out_path = HERE / "output" / f"{slug}.md"

    zot    = zotero.Zotero(ZOTERO_USER_ID, "user", ZOTERO_API_KEY)
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    chars_per_paper = CONTEXT_BUDGET // args.limit
    print(f"\nModel: {CLAUDE_MODEL}  |  context: {CONTEXT_TOKENS:,} tokens")
    print(f"Papers: {args.limit}  |  budget: {chars_per_paper:,} chars/paper (~{chars_per_paper//4:,} tokens)")
    avg_paper_chars = 120_000   # median from library (38 papers, range 9k–353k)
    coverage = min(100, round(chars_per_paper / avg_paper_chars * 100))
    print(f"Expected coverage: ~{coverage}% of a typical paper ({avg_paper_chars//1000}k char median)")

    # ── Step 1: Search ──────────────────────────────────────────────────────────
    if args.collection:
        print(f"\n[1/4] Fetching items from collection {args.collection}…")
        candidates = zot.collection_items(args.collection, itemType="-attachment", limit=100)
        print(f"      Found {len(candidates)} items.")
    else:
        print(f"\n[1/4] Searching Zotero: {args.query!r} (up to {args.candidates} candidates)…")
        candidates = search_zotero(zot, args.query, limit=args.candidates)
        print(f"      Found {len(candidates)} candidates.")
    if not candidates:
        print("No results. Try a broader query or different collection.")
        sys.exit(1)

    # ── Step 2: Re-rank ─────────────────────────────────────────────────────────
    top_n = min(args.limit, len(candidates))
    print(f"\n[2/4] Re-ranking with Claude → selecting top {top_n}…")
    selected_keys = rerank_with_claude(client, args.query, candidates, top_n)
    print(f"      Selected keys: {selected_keys}")

    # ── Step 3: Fetch full texts ────────────────────────────────────────────────
    max_chars = CONTEXT_BUDGET // len(selected_keys)
    print(f"\n[3/4] Fetching full texts (budget: {max_chars:,} chars/paper, cache: {TEXT_CACHE})…")
    papers = []
    for key in selected_keys:
        meta = get_item_metadata(zot, key)
        print(f"  {meta['authors']} ({meta['year']}) — {meta['title'][:65]}")
        text = get_full_text(zot, key, max_chars=max_chars)
        papers.append({**meta, "text": text})

    # ── Step 4: Synthesise ──────────────────────────────────────────────────────
    print(f"\n[4/4] Synthesising literature review…")
    synthesize(client, args.query, papers, out_path)


if __name__ == "__main__":
    main()
