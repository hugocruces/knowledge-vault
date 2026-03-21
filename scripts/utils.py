"""
utils.py — Shared helpers for knowledge-vault scripts.

Exports:
    REPO_ROOT          — Path to the repository root
    MAX_TEXT_CHARS     — Max chars extracted from a PDF for metadata extraction
    ITEM_TYPE_MAP      — Normalised item-type string → Zotero itemType
    strip_json_fences  — Strip accidental markdown ```json fences from LLM output
    extract_pdf_text   — Extract plain text from the first N pages of a PDF
    EXTRACT_PROMPT     — Prompt prefix for Claude bibliographic extraction
    extract_metadata   — Call Claude to extract bibliographic metadata as a dict
    crossref_enrich    — Enrich a metadata dict with CrossRef data in-place
    build_zotero_item  — Build a Zotero API item dict from a metadata dict
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import fitz          # PyMuPDF
import requests
import anthropic

# ── Paths ─────────────────────────────────────────────────────────────────────

# scripts/ lives one level below the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_TEXT_CHARS = 6_000   # characters fed to Claude for metadata extraction

ITEM_TYPE_MAP: dict[str, str] = {
    "journal article":  "journalArticle",
    "book":             "book",
    "book chapter":     "bookSection",
    "book section":     "bookSection",
    "report":           "report",
    "working paper":    "report",
    "thesis":           "thesis",
    "conference paper": "conferencePaper",
    "preprint":         "preprint",
    "other":            "document",
}

# ── JSON helpers ──────────────────────────────────────────────────────────────

def strip_json_fences(raw: str) -> str:
    """Remove accidental markdown ```json … ``` fences from LLM output."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


# ── PDF text extraction ───────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: Path, max_pages: int = 4) -> str:
    """Return the first *max_pages* pages of text, capped at MAX_TEXT_CHARS."""
    doc = fitz.open(str(pdf_path))
    pages = min(max_pages, len(doc))
    text = "\n\n".join(doc[i].get_text() for i in range(pages))
    return text[:MAX_TEXT_CHARS]


# ── Claude: extract bibliographic metadata ────────────────────────────────────

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


def extract_metadata(client: anthropic.Anthropic, text: str, model: str) -> dict:
    """Call Claude to extract bibliographic metadata from PDF text."""
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": EXTRACT_PROMPT + text}],
    )
    raw = strip_json_fences(msg.content[0].text)
    return json.loads(raw)


# ── CrossRef enrichment ───────────────────────────────────────────────────────

def crossref_enrich(meta: dict, crossref_mailto: str = "") -> dict:
    """
    Look up the paper on CrossRef and fill in any missing structured fields.

    Modifies *meta* in-place and also returns it for convenience.
    Only trusts matches with CrossRef score ≥ 50.
    """
    title = meta.get("title") or ""
    first_author = (meta.get("authors") or [{}])[0].get("last", "")

    params: dict = {
        "query.title": title,
        "rows": 1,
        "select": (
            "DOI,title,score,author,issued,container-title,"
            "volume,issue,page,publisher,ISBN,ISSN,abstract,type"
        ),
    }
    if first_author:
        params["query.author"] = first_author
    if crossref_mailto:
        params["mailto"] = crossref_mailto

    try:
        r = requests.get("https://api.crossref.org/works", params=params, timeout=10)
        r.raise_for_status()
        items = r.json()["message"]["items"]
        if not items or items[0].get("score", 0) < 50:
            print("  CrossRef: no confident match.")
            return meta
        hit = items[0]
    except Exception as e:
        print(f"  CrossRef warning: {e}")
        return meta

    def _first(field: str):
        v = hit.get(field)
        return v[0] if isinstance(v, list) else v

    if not meta.get("doi"):
        meta["doi"] = hit.get("DOI")
    if not meta.get("journal"):
        meta["journal"] = _first("container-title")
    if not meta.get("volume"):
        meta["volume"] = str(hit["volume"]) if hit.get("volume") else None
    if not meta.get("issue"):
        meta["issue"] = str(hit["issue"]) if hit.get("issue") else None
    if not meta.get("pages"):
        meta["pages"] = hit.get("page")
    if not meta.get("publisher"):
        meta["publisher"] = hit.get("publisher")
    if not meta.get("isbn"):
        meta["isbn"] = (hit.get("ISBN") or [None])[0]
    if not meta.get("issn"):
        meta["issn"] = (hit.get("ISSN") or [None])[0]
    if not meta.get("abstract"):
        meta["abstract"] = hit.get("abstract")
    cr_authors = hit.get("author", [])
    if cr_authors and not meta.get("authors"):
        meta["authors"] = [
            {"first": a.get("given", ""), "last": a.get("family", "")}
            for a in cr_authors
        ]
    parts = hit.get("issued", {}).get("date-parts", [[None]])[0]
    if parts and parts[0] and not meta.get("year"):
        meta["year"] = str(parts[0])

    print(f"  CrossRef: DOI={meta.get('doi')}  score={hit.get('score', '?')}")
    return meta


# ── Build Zotero item template ────────────────────────────────────────────────

def build_zotero_item(meta: dict, collection_keys: list[str]) -> dict:
    """Build a Zotero API item dict from extracted metadata."""
    item_type = ITEM_TYPE_MAP.get((meta.get("item_type") or "other").lower(), "document")

    creators = []
    for a in (meta.get("authors") or []):
        creators.append({
            "creatorType": "author",
            "firstName": a.get("first", ""),
            "lastName": a.get("last", ""),
        })
    for e in (meta.get("editors") or []):
        creators.append({
            "creatorType": "editor",
            "firstName": e.get("first", ""),
            "lastName": e.get("last", ""),
        })

    item: dict = {
        "itemType":     item_type,
        "title":        meta.get("title") or "",
        "creators":     creators,
        "date":         meta.get("year") or "",
        "abstractNote": meta.get("abstract") or "",
        "language":     meta.get("language") or "",
        "collections":  collection_keys,
    }

    if item_type == "journalArticle":
        item.update({
            "publicationTitle": meta.get("journal") or "",
            "volume":           meta.get("volume") or "",
            "issue":            meta.get("issue") or "",
            "pages":            meta.get("pages") or "",
            "DOI":              meta.get("doi") or "",
            "ISSN":             meta.get("issn") or "",
        })
    elif item_type == "bookSection":
        item.update({
            "bookTitle":  meta.get("book_title") or "",
            "publisher":  meta.get("publisher") or "",
            "place":      meta.get("place") or "",
            "pages":      meta.get("pages") or "",
            "ISBN":       meta.get("isbn") or "",
            "DOI":        meta.get("doi") or "",
        })
    elif item_type == "book":
        item.update({
            "publisher": meta.get("publisher") or "",
            "place":     meta.get("place") or "",
            "ISBN":      meta.get("isbn") or "",
            "DOI":       meta.get("doi") or "",
        })
    elif item_type == "report":
        item.update({
            "institution": meta.get("institution") or meta.get("publisher") or "",
            "place":       meta.get("place") or "",
            "DOI":         meta.get("doi") or "",
            "url":         meta.get("url") or "",
        })

    if "url" not in item and meta.get("url"):
        item["url"] = meta["url"]

    return item
