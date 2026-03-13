#!/home/hugo/.local/share/uv/tools/zotero-mcp-server/bin/python3
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
import textwrap
from pathlib import Path

import shutil
import fitz
import requests
import anthropic
from dotenv import load_dotenv
from pyzotero import zotero

ZOTERO_STORAGE = Path.home() / "Zotero" / "storage"

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID", "")
ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY", "")
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL   = "claude-haiku-4-5-20251001"
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "")
MAX_TEXT_CHARS = 6000

ITEM_TYPE_MAP = {
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

# ── Helpers shared with zotero_import.py ─────────────────────────────────────

def extract_pdf_text(pdf_path: Path, max_pages: int = 4) -> str:
    doc = fitz.open(str(pdf_path))
    pages = min(max_pages, len(doc))
    text = "\n\n".join(doc[i].get_text() for i in range(pages))
    return text[:MAX_TEXT_CHARS]


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

def extract_metadata(client: anthropic.Anthropic, text: str) -> dict:
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": EXTRACT_PROMPT + text}],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


def crossref_enrich(meta: dict) -> dict:
    title = meta.get("title") or ""
    author = (meta.get("authors") or [{}])[0].get("last", "")
    params = {
        "query.title": title,
        "rows": 1,
        "select": "DOI,title,score,author,issued,container-title,volume,issue,page,"
                  "publisher,ISBN,ISSN,abstract,type",
    }
    if author:
        params["query.author"] = author
    if CROSSREF_MAILTO:
        params["mailto"] = CROSSREF_MAILTO
    try:
        r = requests.get("https://api.crossref.org/works", params=params, timeout=10)
        r.raise_for_status()
        items = r.json()["message"]["items"]
        if not items or items[0].get("score", 0) < 50:
            print("  CrossRef: no confident match.")
            return meta
        hit = items[0]

        def cr(field):
            v = hit.get(field)
            return v[0] if isinstance(v, list) else v

        if not meta.get("doi"):        meta["doi"]       = hit.get("DOI")
        if not meta.get("journal"):    meta["journal"]   = cr("container-title")
        if not meta.get("volume"):     meta["volume"]    = str(hit["volume"]) if hit.get("volume") else None
        if not meta.get("issue"):      meta["issue"]     = str(hit["issue"])  if hit.get("issue")  else None
        if not meta.get("pages"):      meta["pages"]     = hit.get("page")
        if not meta.get("publisher"):  meta["publisher"] = hit.get("publisher")
        if not meta.get("isbn"):
            meta["isbn"] = (hit.get("ISBN") or [None])[0]
        if not meta.get("issn"):
            meta["issn"] = (hit.get("ISSN") or [None])[0]
        if not meta.get("abstract"):   meta["abstract"]  = hit.get("abstract")
        cr_authors = hit.get("author", [])
        if cr_authors and not meta.get("authors"):
            meta["authors"] = [{"first": a.get("given",""), "last": a.get("family","")} for a in cr_authors]
        parts = hit.get("issued", {}).get("date-parts", [[None]])[0]
        if parts and parts[0] and not meta.get("year"):
            meta["year"] = str(parts[0])
        print(f"  CrossRef: DOI={meta.get('doi')}  score={hit.get('score','?')}")
    except Exception as e:
        print(f"  CrossRef warning: {e}")
    return meta


def build_item(meta: dict, col_keys: list[str]) -> dict:
    itype = ITEM_TYPE_MAP.get((meta.get("item_type") or "other").lower(), "document")

    creators = []
    for a in (meta.get("authors") or []):
        creators.append({"creatorType": "author",
                         "firstName": a.get("first",""), "lastName": a.get("last","")})
    for e in (meta.get("editors") or []):
        creators.append({"creatorType": "editor",
                         "firstName": e.get("first",""), "lastName": e.get("last","")})

    item = {
        "itemType":     itype,
        "title":        meta.get("title") or "",
        "creators":     creators,
        "date":         meta.get("year") or "",
        "abstractNote": meta.get("abstract") or "",
        "language":     meta.get("language") or "",
        "collections":  col_keys,
    }
    if itype == "journalArticle":
        item.update({"publicationTitle": meta.get("journal") or "",
                     "volume": meta.get("volume") or "", "issue": meta.get("issue") or "",
                     "pages": meta.get("pages") or "", "DOI": meta.get("doi") or "",
                     "ISSN": meta.get("issn") or ""})
    elif itype == "bookSection":
        item.update({"bookTitle": meta.get("book_title") or "",
                     "publisher": meta.get("publisher") or "", "place": meta.get("place") or "",
                     "pages": meta.get("pages") or "", "ISBN": meta.get("isbn") or "",
                     "DOI": meta.get("doi") or ""})
    elif itype == "book":
        item.update({"publisher": meta.get("publisher") or "", "place": meta.get("place") or "",
                     "ISBN": meta.get("isbn") or "", "DOI": meta.get("doi") or ""})
    elif itype == "report":
        item.update({"institution": meta.get("institution") or meta.get("publisher") or "",
                     "place": meta.get("place") or "", "DOI": meta.get("doi") or "",
                     "url": meta.get("url") or ""})
    if "url" not in item and meta.get("url"):
        item["url"] = meta["url"]
    return item


# ── Duplicate detection ───────────────────────────────────────────────────────

def is_duplicate(zot, title: str, doi: str | None) -> bool:
    """Return True if a matching item already exists in the Zotero library."""
    if doi:
        results = zot.items(q=doi, qmode="everything", limit=1)
        if results:
            return True
    if title:
        results = zot.items(q=title[:60], limit=5)
        title_lower = title.lower().strip()
        for r in results:
            existing = r["data"].get("title", "").lower().strip()
            # Simple prefix match — enough to catch exact duplicates
            if existing and (existing in title_lower or title_lower in existing):
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
    """
    Returns: "ok" | "duplicate" | "failed"
    """
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
        meta = extract_metadata(client, text)
    except Exception as e:
        print(f"  ERROR parsing Claude response: {e}")
        return "failed"

    print(f"  → {meta.get('item_type','?')} | {(meta.get('title') or '')[:60]}")
    print(f"    Authors: {[a.get('last','') for a in (meta.get('authors') or [])]}")
    print(f"    Year: {meta.get('year')}  DOI: {meta.get('doi')}")

    # 3. CrossRef
    meta = crossref_enrich(meta)

    if dry_run:
        print("  [dry-run] skipping Zotero write.")
        return "ok"

    # 4. Duplicate check
    if is_duplicate(zot, meta.get("title",""), meta.get("doi")):
        print("  DUPLICATE — skipping.")
        return "duplicate"

    # 5. Create parent item
    zitem = build_item(meta, col_keys)
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

    missing = [k for k, v in {"ANTHROPIC_API_KEY": ANTHROPIC_KEY,
                               "ZOTERO_API_KEY": ZOTERO_API_KEY,
                               "ZOTERO_USER_ID": ZOTERO_USER_ID}.items() if not v]
    if missing:
        print(f"ERROR: missing config: {', '.join(missing)}")
        sys.exit(1)

    # Load manifests
    assignments_path = HERE / "paper_assignments.json"
    if not assignments_path.exists():
        print("ERROR: paper_assignments.json not found. Run analyze_library.py --assign first.")
        sys.exit(1)

    manifest = json.loads(assignments_path.read_text())
    assignments   = manifest["assignments"]          # filename → [collection names]
    excluded      = set(manifest.get("excluded", []))
    col_names_all = manifest["collections"]

    # Progress
    if args.reset and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print("Progress reset.")
    progress = load_progress()
    done_set     = set(progress["done"])
    failed_set   = set(progress["failed"])
    skipped_set  = set(progress["skipped"])

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

    # Build PDF list
    all_pdfs = sorted(folder.glob("*.pdf"))
    pdfs = [p for p in all_pdfs
            if p.name not in excluded
            and p.name not in done_set
            and p.name not in skipped_set]

    total     = len(all_pdfs)
    excluded_n = len([p for p in all_pdfs if p.name in excluded])
    already   = len(done_set) + len(skipped_set)
    print(f"\n{total} PDFs total | {excluded_n} excluded | {already} already done | {len(pdfs)} to process\n")

    counts = {"ok": 0, "duplicate": 0, "failed": 0}

    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf.name}")

        # Resolve collection keys
        col_names_for_paper = assignments.get(pdf.name, [])
        col_keys = [name_to_key[n] for n in col_names_for_paper if n in name_to_key]
        print(f"  Collections: {col_names_for_paper or '(none)'}")

        result = import_pdf(pdf, col_keys, zot, client, args.dry_run)
        counts[result] += 1

        if result == "ok":
            progress["done"].append(pdf.name)
        elif result == "duplicate":
            progress["skipped"].append(pdf.name)
        else:
            progress["failed"].append(pdf.name)

        if not args.dry_run:
            save_progress(progress)

        time.sleep(0.3)

    print(f"\n{'='*60}")
    print(f"Done. imported={counts['ok']}  duplicates={counts['duplicate']}  failed={counts['failed']}")
    if progress["failed"]:
        print(f"\nFailed files saved in import_progress.json — fix and re-run to retry.")


if __name__ == "__main__":
    main()
