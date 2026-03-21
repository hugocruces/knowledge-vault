#!/usr/bin/env python3
"""
import_library.py — Phase 2: import PDFs into Zotero using paper_assignments.json.

Features:
  - Creates any new Zotero collections defined in proposed_collections.json
  - Assigns each paper to multiple collections as per paper_assignments.json
  - Extracts metadata via Claude + CrossRef enrichment
  - Duplicate detection (skips papers already in Zotero by title)
  - Resumable: progress saved to import_progress.json after each paper

Usage:
    import_library.py /path/to/pdfs
    import_library.py /path/to/pdfs --dry-run     # no Zotero writes
    import_library.py /path/to/pdfs --reset       # clear progress and restart
"""

import os
import sys
import json
import time
import argparse
import shutil
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pyzotero import zotero

from utils import (
    REPO_ROOT,
    extract_pdf_text,
    extract_metadata,
    crossref_enrich,
    build_zotero_item,
)

ZOTERO_STORAGE = Path.home() / "Zotero" / "storage"

HERE = Path(__file__).resolve().parent
load_dotenv(REPO_ROOT / ".env")

ZOTERO_USER_ID  = os.getenv("ZOTERO_USER_ID", "")
ZOTERO_API_KEY  = os.getenv("ZOTERO_API_KEY", "")
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "")
CLAUDE_MODEL    = "claude-haiku-4-5-20251001"


# ── Duplicate detection ───────────────────────────────────────────────────────

def _normalize(t: str) -> list[str]:
    return "".join(c.lower() for c in t if c.isalnum() or c.isspace()).split()


def is_duplicate(zot, title: str, doi: str | None) -> bool:
    """Return True if a matching item already exists in the Zotero library."""
    # DOI match is authoritative
    if doi:
        results = zot.items(q=doi, qmode="everything", limit=3)
        for r in results:
            if doi.lower() in (r["data"].get("DOI") or "").lower():
                return True
    # Title match: require ≥90% word overlap
    if title:
        results = zot.items(q=title[:80], limit=5)
        words_a = set(_normalize(title))
        if not words_a:
            return False
        for r in results:
            existing = r["data"].get("title", "")
            if not existing:
                continue
            words_b = set(_normalize(existing))
            overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
            if overlap >= 0.9:
                return True
    return False


# ── Collection management ─────────────────────────────────────────────────────

def ensure_collections(zot, proposed_names: list[str]) -> dict[str, str]:
    """
    Create any missing collections. Returns name → key mapping for all
    collections in proposed_names.
    """
    existing = {c["data"]["name"]: c["key"] for c in zot.collections()}
    name_to_key = {}
    for name in proposed_names:
        if name in existing:
            name_to_key[name] = existing[name]
        else:
            resp = zot.create_collections([{"name": name, "parentCollection": False}])
            new_key = list(resp["success"].values())[0]
            name_to_key[name] = new_key
            print(f"  Created collection: {name} ({new_key})")
    return name_to_key


# ── Progress tracking ─────────────────────────────────────────────────────────

PROGRESS_FILE = HERE / "import_progress.json"


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"done": [], "failed": [], "skipped": []}


def save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


# ── Per-paper import ──────────────────────────────────────────────────────────

def import_pdf(pdf_path: Path, col_keys: list[str], zot,
               client: anthropic.Anthropic, dry_run: bool) -> str:
    """Returns: "ok" | "duplicate" | "failed" """
    # 1. Extract text
    try:
        text = extract_pdf_text(pdf_path)
    except Exception as e:
        print(f"  ERROR extracting text: {e}")
        return "failed"

    if not text.strip():
        print("  ERROR: no text extracted (scanned PDF?)")
        return "failed"

    # 2. Claude metadata
    try:
        meta = extract_metadata(client, text, CLAUDE_MODEL)
    except Exception as e:
        print(f"  ERROR parsing Claude response: {e}")
        return "failed"

    print(f"  → {meta.get('item_type','?')} | {(meta.get('title') or '')[:60]}")
    print(f"    Authors: {[a.get('last','') for a in (meta.get('authors') or [])]}")
    print(f"    Year: {meta.get('year')}  DOI: {meta.get('doi')}")

    # 3. CrossRef
    meta = crossref_enrich(meta, CROSSREF_MAILTO)

    if dry_run:
        print("  [dry-run] skipping Zotero write.")
        return "ok"

    # 4. Duplicate check
    if is_duplicate(zot, meta.get("title", ""), meta.get("doi")):
        print("  DUPLICATE — skipping.")
        return "duplicate"

    # 5. Create parent item
    zitem = build_zotero_item(meta, col_keys)
    try:
        resp = zot.create_items([zitem])
        if not resp.get("success"):
            print(f"  ERROR creating item: {resp}")
            return "failed"
        parent_key = list(resp["success"].values())[0]
    except Exception as e:
        print(f"  ERROR creating item: {e}")
        return "failed"

    # 6. Create attachment item and copy PDF into Zotero local storage
    try:
        basename = pdf_path.name
        attachment_template = [{
            "itemType":    "attachment",
            "linkMode":    "imported_file",
            "title":       basename,
            "contentType": "application/pdf",
            "filename":    basename,
            "parentItem":  parent_key,
            "collections": [],
            "tags": [],
        }]
        att_resp = zot.create_items(attachment_template)
        if not att_resp.get("success"):
            print(f"  WARNING: item created ({parent_key}) but attachment metadata failed.")
            return "ok"
        att_key = list(att_resp["success"].values())[0]

        # Copy the PDF into ~/Zotero/storage/{att_key}/
        dest_dir = ZOTERO_STORAGE / att_key
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, dest_dir / basename)
        print(f"  Created {parent_key}, PDF copied to storage/{att_key}/")
    except Exception as e:
        print(f"  WARNING: item created ({parent_key}) but attachment failed: {e}")

    return "ok"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2: import PDFs into Zotero.")
    parser.add_argument("folder", help="Folder containing PDFs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset", action="store_true",
                        help="Clear saved progress and start from scratch")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()

    missing = [k for k, v in {
        "ANTHROPIC_API_KEY": ANTHROPIC_KEY,
        "ZOTERO_API_KEY":    ZOTERO_API_KEY,
        "ZOTERO_USER_ID":    ZOTERO_USER_ID,
    }.items() if not v]
    if missing:
        print(f"ERROR: missing config: {', '.join(missing)}")
        sys.exit(1)

    # Load manifests
    assignments_path = HERE / "paper_assignments.json"
    if not assignments_path.exists():
        print("ERROR: paper_assignments.json not found. Run analyze_library.py --assign first.")
        sys.exit(1)

    manifest      = json.loads(assignments_path.read_text())
    assignments   = manifest["assignments"]      # filename → [collection names]
    excluded      = set(manifest.get("excluded", []))
    col_names_all = manifest["collections"]

    # Progress
    if args.reset and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print("Progress reset.")
    progress    = load_progress()
    done_set    = set(progress["done"])
    failed_set  = set(progress["failed"])
    skipped_set = set(progress["skipped"])

    # Init clients
    zot    = zotero.Zotero(ZOTERO_USER_ID, "user", ZOTERO_API_KEY)
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    # Ensure all collections exist in Zotero
    print("Ensuring collections exist in Zotero…")
    if not args.dry_run:
        name_to_key = ensure_collections(zot, col_names_all)
    else:
        name_to_key = {n: f"DRYRUN-{i}" for i, n in enumerate(col_names_all)}
    print(f"  {len(name_to_key)} collections ready.")

    # Build PDF list: always retry failed, skip done/skipped
    all_pdfs = sorted(folder.glob("*.pdf"))
    pdfs = [p for p in all_pdfs
            if p.name not in excluded
            and (p.name in failed_set
                 or (p.name not in done_set and p.name not in skipped_set))]

    total      = len(all_pdfs)
    excluded_n = len([p for p in all_pdfs if p.name in excluded])
    already    = len([p for p in all_pdfs
                      if p.name not in excluded
                      and p.name not in failed_set
                      and (p.name in done_set or p.name in skipped_set)])
    print(f"\n{total} PDFs total | {excluded_n} excluded | {already} already done | {len(pdfs)} to process\n")

    counts = {"ok": 0, "duplicate": 0, "failed": 0}

    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf.name}")

        col_names_for_paper = assignments.get(pdf.name, [])
        col_keys = [name_to_key[n] for n in col_names_for_paper if n in name_to_key]
        print(f"  Collections: {col_names_for_paper or '(none)'}")

        result = import_pdf(pdf, col_keys, zot, client, args.dry_run)
        counts[result] += 1

        if result == "ok":
            progress["done"].append(pdf.name)
            if pdf.name in progress["failed"]:
                progress["failed"].remove(pdf.name)
        elif result == "duplicate":
            progress["skipped"].append(pdf.name)
            if pdf.name in progress["failed"]:
                progress["failed"].remove(pdf.name)
        else:
            if pdf.name not in progress["failed"]:
                progress["failed"].append(pdf.name)

        if not args.dry_run:
            save_progress(progress)

        time.sleep(0.3)

    print(f"\n{'='*60}")
    print(f"Done. imported={counts['ok']}  duplicates={counts['duplicate']}  failed={counts['failed']}")
    if progress["failed"]:
        print("Failed files saved in import_progress.json — fix and re-run to retry.")


if __name__ == "__main__":
    main()
