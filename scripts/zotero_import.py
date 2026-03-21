#!/usr/bin/env python3
"""
zotero_import.py — Bulk PDF importer for Zotero with LLM metadata extraction
             and CrossRef enrichment.

Usage:
    python3 zotero_import.py [folder]          # defaults to current directory
    python3 zotero_import.py /path/to/pdfs

Requirements:
    anthropic, pyzotero, pymupdf, requests, python-dotenv
    (see requirements.txt)

Config: set env vars or copy .env.example to .env at the repo root.
"""

import os
import sys
import time
import argparse
import textwrap
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pyzotero import zotero

# Ensure scripts/ is on sys.path so utils.py is importable regardless of
# the working directory or whether the script is invoked via a symlink.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    REPO_ROOT,
    ITEM_TYPE_MAP,
    extract_pdf_text,
    extract_metadata,
    crossref_enrich,
    build_zotero_item,
)

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv(REPO_ROOT / ".env")

ZOTERO_USER_ID  = os.getenv("ZOTERO_USER_ID", "")
ZOTERO_API_KEY  = os.getenv("ZOTERO_API_KEY", "")
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "")
CLAUDE_MODEL    = "claude-haiku-4-5-20251001"    # fast + cheap for extraction


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
        meta = extract_metadata(client, text, CLAUDE_MODEL)
    except Exception as e:
        print(f"  ERROR parsing Claude response: {e}")
        return False

    print(f"  → {meta.get('item_type','?')} | {meta.get('title','?')[:60]}")
    print(f"    Authors: {meta.get('authors')}")
    print(f"    Year: {meta.get('year')} | DOI: {meta.get('doi')}")

    # 3. CrossRef enrichment
    meta = crossref_enrich(meta, CROSSREF_MAILTO)

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
        print("  PDF attached OK")
    else:
        print(f"  WARNING: attachment may have failed: {attach}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Bulk import PDFs into Zotero with LLM metadata.")
    parser.add_argument("folder", nargs="?", default=".", help="Folder containing PDFs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract metadata only, don't write to Zotero")
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

    print(f"Found {len(pdfs)} PDF(s) in {folder}")

    # Init clients
    zot    = zotero.Zotero(ZOTERO_USER_ID, "user", ZOTERO_API_KEY)
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

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
