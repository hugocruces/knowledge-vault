#!/usr/bin/env python3
"""
zotero_import.py — Bulk PDF importer for Zotero with LLM metadata extraction
             and CrossRef/Semantic Scholar enrichment.

Usage:
    python3 zotero_import.py [folder]          # defaults to current directory
    python3 zotero_import.py /path/to/pdfs

Requirements (all available in the zotero-mcp-server env):
    anthropic, pyzotero, pymupdf, requests

Config: set env vars or edit DEFAULTS below.
"""

import os
import sys
import json
import time
import argparse
import textwrap
from pathlib import Path

import fitz          # PyMuPDF
import requests
import anthropic
from dotenv import load_dotenv
from pyzotero import zotero

# ── Config ────────────────────────────────────────────────────────────────────

# Load .env from the same directory as this script, then fall back to env vars
load_dotenv(Path(__file__).resolve().parent / ".env")

ZOTERO_USER_ID  = os.getenv("ZOTERO_USER_ID", "")
ZOTERO_API_KEY  = os.getenv("ZOTERO_API_KEY", "")
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL    = "claude-haiku-4-5-20251001"           # fast + cheap for extraction

CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "")     # optional but polite
MAX_TEXT_CHARS  = 6000                                  # chars fed to Claude

# ── Zotero item-type map ──────────────────────────────────────────────────────

ITEM_TYPE_MAP = {
    "journal article": "journalArticle",
    "book":            "book",
    "book chapter":    "bookSection",
    "book section":    "bookSection",
    "report":          "report",
    "working paper":   "report",
    "thesis":          "thesis",
    "conference paper":"conferencePaper",
    "preprint":        "preprint",
    "other":           "document",
}

# ── PDF text extraction ───────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: Path, max_pages: int = 4) -> str:
    doc = fitz.open(str(pdf_path))
    pages = min(max_pages, len(doc))
    text = "\n\n".join(doc[i].get_text() for i in range(pages))
    return text[:MAX_TEXT_CHARS]


# ── Claude: extract metadata ──────────────────────────────────────────────────

EXTRACT_PROMPT = """\
You are a bibliographic metadata extractor. Given the first few pages of a PDF, \
return a JSON object with the following fields (use null if unknown):

{
  "title": "...",
  "item_type": "journal article | book | book chapter | report | working paper | \
thesis | conference paper | preprint | other",
  "authors": [{"first": "...", "last": "..."}],
  "editors": [{"first": "...", "last": "..."}],
  "year": "YYYY or null",
  "journal": "journal name or null",
  "volume": "...",
  "issue": "...",
  "pages": "...",
  "publisher": "...",
  "place": "city or null",
  "institution": "institution/org name for reports or null",
  "book_title": "for chapters: title of the book or null",
  "isbn": "...",
  "issn": "...",
  "doi": "...",
  "url": "...",
  "language": "ISO 639-1 code, e.g. en / es / fr",
  "abstract": "short abstract if present, else null"
}

Return ONLY valid JSON, no markdown fences, no commentary.

PDF text:
"""

def extract_metadata_with_claude(client: anthropic.Anthropic, text: str) -> dict:
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": EXTRACT_PROMPT + text}],
    )
    raw = msg.content[0].text.strip()
    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


# ── Claude: classify into collection ─────────────────────────────────────────

def classify_collection(client: anthropic.Anthropic, meta: dict,
                         collections: list[dict], text: str = "") -> list[str]:
    col_list = "\n".join(f'- "{c["name"]}" (key: {c["key"]})' for c in collections)
    snippet = text[:1500] if text else ""
    prompt = textwrap.dedent(f"""\
        You are helping organise an academic library. A paper may belong to MORE THAN ONE \
        collection — assign all that genuinely apply. Reply "none" only if it fits nowhere.

        Collections:
        {col_list}

        Paper metadata:
        Title: {meta.get("title")}
        Type: {meta.get("item_type")}
        Authors: {meta.get("authors")}
        Year: {meta.get("year")}
        Abstract: {meta.get("abstract") or "(none)"}

        Opening text of PDF:
        {snippet}

        Reply with ONLY a comma-separated list of collection keys (e.g. 6IJGA8RL,FM3KKKKY) \
        or the word "none". No spaces, no explanation.
    """)
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=60,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = msg.content[0].text.strip().strip('"')
    valid_keys = {c["key"] for c in collections}
    if answer.lower() == "none":
        return []
    return [k.strip() for k in answer.split(",") if k.strip() in valid_keys]


# ── CrossRef enrichment ───────────────────────────────────────────────────────

def crossref_lookup(title: str, first_author_last: str = "") -> dict | None:
    params = {
        "query.title": title,
        "rows": 1,
        "select": "DOI,title,score,author,issued,container-title,volume,issue,page,"
                  "publisher,ISBN,ISSN,abstract,type",
    }
    if first_author_last:
        params["query.author"] = first_author_last
    if CROSSREF_MAILTO:
        params["mailto"] = CROSSREF_MAILTO

    try:
        r = requests.get("https://api.crossref.org/works", params=params, timeout=10)
        r.raise_for_status()
        items = r.json()["message"]["items"]
        if not items:
            return None
        hit = items[0]
        score = hit.get("score", 0)
        # Only trust high-confidence matches
        if score < 50:
            return None
        return hit
    except Exception as e:
        print(f"  [crossref] warning: {e}")
        return None


def semantic_scholar_lookup(title: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            r = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": title,
                    "limit": 1,
                    "fields": "title,authors,year,abstract,externalIds,venue,publicationDate",
                },
                timeout=10,
            )
            if r.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"  [semantic scholar] rate limited, waiting {wait}s…")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json().get("data", [])
            return data[0] if data else None
        except Exception as e:
            print(f"  [semantic scholar] warning: {e}")
            return None
    return None


def enrich_with_crossref(meta: dict) -> dict:
    """Merge CrossRef data into meta, preferring CrossRef for structured fields."""
    title = meta.get("title") or ""
    first_author = (meta.get("authors") or [{}])[0].get("last", "")

    print(f"  Querying CrossRef for: {title[:60]}…")
    hit = crossref_lookup(title, first_author)

    if not hit:
        print("  CrossRef: no confident match.")
        return meta

    # Merge CrossRef fields
    def cr_str(field):
        v = hit.get(field)
        if isinstance(v, list):
            return v[0] if v else None
        return v

    if not meta.get("doi"):
        meta["doi"] = hit.get("DOI")
    if not meta.get("journal"):
        meta["journal"] = cr_str("container-title")
    if not meta.get("volume"):
        meta["volume"] = str(hit["volume"]) if hit.get("volume") else None
    if not meta.get("issue"):
        meta["issue"] = str(hit["issue"]) if hit.get("issue") else None
    if not meta.get("pages"):
        meta["pages"] = hit.get("page")
    if not meta.get("publisher"):
        meta["publisher"] = hit.get("publisher")
    if not meta.get("isbn"):
        isbns = hit.get("ISBN", [])
        meta["isbn"] = isbns[0] if isbns else None
    if not meta.get("issn"):
        issns = hit.get("ISSN", [])
        meta["issn"] = issns[0] if issns else None
    if not meta.get("abstract"):
        meta["abstract"] = hit.get("abstract")
    # CrossRef authors (more structured)
    cr_authors = hit.get("author", [])
    if cr_authors and not meta.get("authors"):
        meta["authors"] = [
            {"first": a.get("given", ""), "last": a.get("family", "")}
            for a in cr_authors
        ]
    issued = hit.get("issued", {}).get("date-parts", [[None]])[0]
    if issued and issued[0] and not meta.get("year"):
        meta["year"] = str(issued[0])

    print(f"  CrossRef match: DOI={meta.get('doi')} score={hit.get('score', '?')}")
    return meta


# ── Build Zotero item template ────────────────────────────────────────────────

def build_zotero_item(meta: dict, collection_keys: list[str]) -> dict:
    item_type = ITEM_TYPE_MAP.get((meta.get("item_type") or "other").lower(), "document")

    def creators():
        result = []
        for a in (meta.get("authors") or []):
            result.append({
                "creatorType": "author",
                "firstName": a.get("first", ""),
                "lastName": a.get("last", ""),
            })
        for e in (meta.get("editors") or []):
            result.append({
                "creatorType": "editor",
                "firstName": e.get("first", ""),
                "lastName": e.get("last", ""),
            })
        return result

    item = {
        "itemType": item_type,
        "title": meta.get("title") or "",
        "creators": creators(),
        "date": meta.get("year") or "",
        "abstractNote": meta.get("abstract") or "",
        "language": meta.get("language") or "",
        "collections": collection_keys,
    }

    # Type-specific fields
    if item_type == "journalArticle":
        item.update({
            "publicationTitle": meta.get("journal") or "",
            "volume": meta.get("volume") or "",
            "issue": meta.get("issue") or "",
            "pages": meta.get("pages") or "",
            "DOI": meta.get("doi") or "",
            "ISSN": meta.get("issn") or "",
        })
    elif item_type in ("bookSection",):
        item.update({
            "bookTitle": meta.get("book_title") or "",
            "publisher": meta.get("publisher") or "",
            "place": meta.get("place") or "",
            "pages": meta.get("pages") or "",
            "ISBN": meta.get("isbn") or "",
            "DOI": meta.get("doi") or "",
        })
    elif item_type in ("book",):
        item.update({
            "publisher": meta.get("publisher") or "",
            "place": meta.get("place") or "",
            "ISBN": meta.get("isbn") or "",
            "DOI": meta.get("doi") or "",
        })
    elif item_type in ("report",):
        item.update({
            "institution": meta.get("institution") or meta.get("publisher") or "",
            "place": meta.get("place") or "",
            "DOI": meta.get("doi") or "",
            "url": meta.get("url") or "",
        })

    # URL fallback
    if "url" not in item and meta.get("url"):
        item["url"] = meta["url"]

    return item


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_pdf(pdf_path: Path, zot, client: anthropic.Anthropic,
                collections: list[dict], dry_run: bool = False) -> bool:
    print(f"\n{'─'*60}")
    print(f"Processing: {pdf_path.name}")

    # 1. Extract text
    print("  Extracting text…")
    text = extract_pdf_text(pdf_path)
    if not text.strip():
        print("  ERROR: could not extract text (scanned PDF?)")
        return False

    # 2. Claude metadata extraction
    print("  Extracting metadata with Claude…")
    try:
        meta = extract_metadata_with_claude(client, text)
    except Exception as e:
        print(f"  ERROR parsing Claude response: {e}")
        return False

    print(f"  → {meta.get('item_type','?')} | {meta.get('title','?')[:60]}")
    print(f"    Authors: {meta.get('authors')}")
    print(f"    Year: {meta.get('year')} | DOI: {meta.get('doi')}")

    # 3. CrossRef/Semantic Scholar enrichment
    meta = enrich_with_crossref(meta)

    # 4. Classify collection
    print("  Classifying collection…")
    col_keys = classify_collection(client, meta, collections, text)
    col_names = [c["name"] for c in collections if c["key"] in col_keys]
    print(f"  → Collections: {', '.join(col_names) or 'none'}")

    if dry_run:
        print("  [dry-run] Skipping Zotero write.")
        return True

    # 5. Create Zotero parent item
    zotero_item = build_zotero_item(meta, col_keys)
    print("  Creating Zotero item…")
    resp = zot.create_items([zotero_item])
    if not resp.get("success"):
        print(f"  ERROR creating item: {resp}")
        return False
    parent_key = list(resp["success"].values())[0]
    print(f"  Created parent: {parent_key}")

    # 6. Upload PDF as child attachment
    print("  Uploading PDF…")
    attach = zot.attachment_simple([str(pdf_path)], parentid=parent_key)
    if attach.get("success") or attach.get("unchanged"):
        print(f"  PDF attached OK")
    else:
        print(f"  WARNING: attachment may have failed: {attach}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Bulk import PDFs into Zotero with LLM metadata.")
    parser.add_argument("folder", nargs="?", default=".", help="Folder containing PDFs")
    parser.add_argument("--dry-run", action="store_true", help="Extract metadata only, don't write to Zotero")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {folder}")
        sys.exit(1)

    missing = [k for k, v in {
        "ANTHROPIC_API_KEY": ANTHROPIC_KEY,
        "ZOTERO_API_KEY":    ZOTERO_API_KEY,
        "ZOTERO_USER_ID":    ZOTERO_USER_ID,
    }.items() if not v]
    if missing:
        print(f"ERROR: missing config vars: {', '.join(missing)}")
        print("Set them in .env or as environment variables.")
        sys.exit(1)
    api_key = ANTHROPIC_KEY

    print(f"Found {len(pdfs)} PDF(s) in {folder}")

    # Init clients
    zot = zotero.Zotero(ZOTERO_USER_ID, "user", ZOTERO_API_KEY)
    client = anthropic.Anthropic(api_key=api_key)

    # Fetch collections once
    raw_cols = zot.collections()
    collections = [
        {"key": c["key"], "name": c["data"]["name"]}
        for c in raw_cols
    ]
    print(f"Collections: {[c['name'] for c in collections]}")

    results = {"ok": [], "failed": []}
    for pdf in pdfs:
        ok = process_pdf(pdf, zot, client, collections, dry_run=args.dry_run)
        (results["ok"] if ok else results["failed"]).append(pdf.name)
        time.sleep(0.5)   # be polite to APIs

    print(f"\n{'='*60}")
    print(f"Done. {len(results['ok'])} imported, {len(results['failed'])} failed.")
    if results["failed"]:
        print("Failed:", results["failed"])


if __name__ == "__main__":
    main()
