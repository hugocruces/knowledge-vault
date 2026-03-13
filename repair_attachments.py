#!/home/hugo/.local/share/uv/tools/zotero-mcp-server/bin/python3
"""
repair_attachments.py — Copy missing PDFs into Zotero local storage.

attachment_simple() registers attachments in the API but doesn't copy files
to ~/Zotero/storage/{key}/. This script finds every imported_file attachment
whose file is missing locally, matches it to the original PDF, and copies it.

Usage:
    repair_attachments.py /path/to/source/pdf/folder
    repair_attachments.py /path/to/source/pdf/folder --dry-run
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from dotenv import load_dotenv
from pyzotero import zotero

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID", "")
ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY", "")
ZOTERO_STORAGE = Path.home() / "Zotero" / "storage"


def stem(name: str) -> str:
    """Lowercase stem for fuzzy matching."""
    return Path(name).stem.lower()


def find_source(filename: str, title: str, source_dir: Path) -> Path | None:
    """
    Try to find the original PDF in source_dir by:
    1. Exact filename match
    2. Stem match (ignores extension case differences)
    3. Title substring match against all PDF stems
    """
    # 1. Exact
    exact = source_dir / filename
    if exact.exists():
        return exact

    # 2. Stem match
    fn_stem = stem(filename)
    for pdf in source_dir.glob("*.pdf"):
        if stem(pdf.name) == fn_stem:
            return pdf

    # 3. Title substring (if title given and meaningful)
    if title and len(title) > 10:
        title_lower = title.lower()
        for pdf in source_dir.glob("*.pdf"):
            if title_lower[:40] in stem(pdf.name) or stem(pdf.name)[:40] in title_lower:
                return pdf

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_folder", help="Folder containing the original PDFs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_folder).expanduser().resolve()
    if not source_dir.exists():
        print(f"ERROR: {source_dir} does not exist")
        sys.exit(1)

    zot = zotero.Zotero(ZOTERO_USER_ID, "user", ZOTERO_API_KEY)

    print("Fetching all attachment items from Zotero…")
    attachments = zot.everything(zot.items(itemType="attachment"))
    pdf_attachments = [
        a for a in attachments
        if a["data"].get("linkMode") == "imported_file"
        and a["data"].get("contentType") == "application/pdf"
    ]
    print(f"{len(pdf_attachments)} imported_file PDF attachments found.")

    missing = skipped = fixed = errors = 0

    for att in pdf_attachments:
        data   = att["data"]
        key    = att["key"]
        stored_filename = data.get("filename", "") or data.get("title", "")
        title  = data.get("title", "")

        # The correct local path
        dest_dir  = ZOTERO_STORAGE / key
        dest_file = dest_dir / Path(stored_filename).name

        if dest_file.exists():
            skipped += 1
            continue

        missing += 1
        source = find_source(stored_filename, title, source_dir)

        if not source:
            print(f"  [{key}] NOT FOUND in source: {stored_filename[:70]}")
            errors += 1
            continue

        correct_name = source.name
        dest_file    = dest_dir / correct_name

        print(f"  [{key}] {correct_name[:65]}")

        if not args.dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest_file)

            # Also fix the filename metadata in Zotero if it was mangled
            if data.get("filename") != correct_name or data.get("title") != correct_name:
                data["filename"] = correct_name
                data["title"]    = correct_name
                try:
                    zot.update_item(att)
                except Exception as e:
                    print(f"    (metadata update failed: {e})")

        fixed += 1

    print(f"\nDone.  missing={missing}  copied={fixed}  already_ok={skipped}  not_found={errors - (missing - fixed)}")
    if args.dry_run:
        print("(dry-run: no files were copied)")


if __name__ == "__main__":
    main()
