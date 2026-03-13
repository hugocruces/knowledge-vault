#!/home/hugo/.local/share/uv/tools/zotero-mcp-server/bin/python3
"""
analyze_library.py — Phase 1: propose and assign a Zotero collection taxonomy.

Step 1 — propose taxonomy:
    analyze_library.py /path/to/pdfs [--existing-collections]
    → writes proposed_collections.json  (review and edit this)

Step 2 — assign every paper to collections:
    analyze_library.py /path/to/pdfs --assign
    → reads proposed_collections.json, writes paper_assignments.json  (review and edit this)

Then proceed to Phase 2 (import).
"""

import sys
import json
import argparse
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pyzotero import zotero
import os

load_dotenv(Path(__file__).resolve().parent / ".env")

ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID", "")
ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY", "")
CLAUDE_MODEL   = "claude-sonnet-4-6"   # better reasoning for taxonomy task


def parse_filename(name: str) -> dict:
    """Extract author / year / title from 'Author - Year - Title.pdf' pattern."""
    stem = Path(name).stem
    parts = [p.strip() for p in stem.split(" - ", 2)]
    if len(parts) == 3:
        return {"author": parts[0], "year": parts[1], "title": parts[2]}
    if len(parts) == 2:
        # Could be Author - Title or Year - Title
        return {"author": parts[0], "year": "", "title": parts[1]}
    return {"author": "", "year": "", "title": stem}


TAXONOMY_PROMPT = """\
You are helping organise an academic research library of {n} papers.
Below is the full list of papers (author, year, title) parsed from their filenames.

{existing}

Your task: propose a clean collection taxonomy — between 8 and 15 collections — \
that covers the whole library. Prefer thematic clusters over methodological ones, \
unless a methods cluster is genuinely large and coherent.

Important: a paper may belong to MORE THAN ONE collection. For example, a paper on \
intergenerational wealth transmission naturally fits both "Intergenerational Mobility" \
and "Wealth Inequality & Inheritance". Assign multiple collections wherever it makes sense.

For each proposed collection provide:
- A short name (2–4 words, suitable as a Zotero collection label)
- A one-sentence description of what belongs there
- A list of 5–8 representative paper titles from the list (exactly as given)
- An estimated count of how many of the {n} papers would fall in it (counting cross-listed papers multiple times)

Then list any papers that genuinely don't fit any collection (true outliers).

Respond in this JSON format:
{{
  "collections": [
    {{
      "name": "...",
      "description": "...",
      "representative_titles": ["...", "..."],
      "estimated_count": 12
    }}
  ],
  "outliers": ["title1", "title2"]
}}

Return ONLY valid JSON, no markdown fences.

Paper list:
{papers}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing PDFs")
    parser.add_argument("--existing-collections", action="store_true",
                        help="Fetch and show current Zotero collections")
    parser.add_argument("--assign", action="store_true",
                        help="Step 2: assign every paper to collections using proposed_collections.json")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {folder}")
        sys.exit(1)

    if not ANTHROPIC_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set in .env")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    if args.assign:
        run_assignment(folder, client)
        return

    print(f"Found {len(pdfs)} PDFs in {folder}")

    # Parse filenames
    parsed = [parse_filename(p.name) for p in pdfs]
    paper_lines = "\n".join(
        f"{i+1}. [{p['year'] or '?'}] {p['author']} — {p['title']}"
        for i, p in enumerate(parsed)
    )

    # Optionally fetch existing Zotero collections
    existing_block = ""
    if args.existing_collections and ZOTERO_USER_ID and ZOTERO_API_KEY:
        zot = zotero.Zotero(ZOTERO_USER_ID, "user", ZOTERO_API_KEY)
        cols = zot.collections()
        names = [c["data"]["name"] for c in cols]
        existing_block = (
            f"The library already has these Zotero collections: {', '.join(names)}.\n"
            "You may reuse, merge, or extend them, but propose whatever structure best fits the new papers.\n\n"
        )

    prompt = TAXONOMY_PROMPT.format(
        n=len(pdfs),
        papers=paper_lines,
        existing=existing_block,
    )

    print("Sending to Claude for taxonomy analysis…\n")
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        print("ERROR: Claude returned invalid JSON:")
        print(raw)
        sys.exit(1)

    # Pretty-print the proposal
    collections = result.get("collections", [])
    outliers    = result.get("outliers", [])
    total_assigned = sum(c.get("estimated_count", 0) for c in collections)

    print("=" * 65)
    print(f"PROPOSED COLLECTION TAXONOMY  ({len(collections)} collections)")
    print("=" * 65)

    for i, col in enumerate(collections, 1):
        print(f"\n{i:>2}. {col['name']}  (~{col.get('estimated_count', '?')} papers)")
        print(f"    {col['description']}")
        print("    Examples:")
        for t in col.get("representative_titles", []):
            print(f"      · {t}")

    if outliers:
        print(f"\n── Outliers ({len(outliers)} papers with no clear home) ──")
        for t in outliers:
            print(f"  · {t}")

    print(f"\n── Coverage: ~{total_assigned} / {len(pdfs)} papers assigned ──")

    # Save full JSON for use in Phase 2
    out_path = Path(__file__).resolve().parent / "proposed_collections.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nFull proposal saved to: {out_path}")
    print("Review it, edit if needed, then we'll use it in Phase 2 to create the collections and import.")


ASSIGN_PROMPT = """\
You are organising an academic library. Below are the approved collections and a numbered \
list of papers. Assign each paper to one or more collections. A paper MUST be assigned to \
every collection it genuinely fits — cross-listing is expected and encouraged. \
Only leave a paper unassigned if it truly fits none.

Collections:
{collections}

Papers:
{papers}

Return a JSON object mapping each paper number (as a string) to a list of collection names \
(exactly as written above). Papers with no fit get an empty list [].

Example:
{{
  "1": ["Inequality of Opportunity", "Intergenerational Mobility"],
  "2": ["Wealth Inequality & Inheritance"],
  "3": []
}}

Return ONLY valid JSON, no markdown fences.
"""

BATCH_SIZE = 80   # papers per Claude call — keeps response well within token limits


def run_assignment(folder: Path, client: anthropic.Anthropic) -> None:
    here = Path(__file__).resolve().parent
    taxonomy_path = here / "proposed_collections.json"
    if not taxonomy_path.exists():
        print("ERROR: proposed_collections.json not found. Run without --assign first.")
        sys.exit(1)

    taxonomy = json.loads(taxonomy_path.read_text())
    collections = taxonomy.get("collections", [])
    exclude_files = set(taxonomy.get("exclude", []))
    col_names = [c["name"] for c in collections]

    col_block = "\n".join(f'- {c["name"]}: {c["description"]}' for c in collections)

    pdfs = sorted(folder.glob("*.pdf"))
    pdfs = [p for p in pdfs if p.name not in exclude_files]
    print(f"Assigning {len(pdfs)} papers to {len(col_names)} collections (excluding {len(exclude_files)} files)…")

    # Split into batches
    batches = [pdfs[i:i + BATCH_SIZE] for i in range(0, len(pdfs), BATCH_SIZE)]
    all_assignments = {}   # filename → [collection names]

    for b_idx, batch in enumerate(batches):
        print(f"  Batch {b_idx + 1}/{len(batches)} ({len(batch)} papers)…")
        parsed = [parse_filename(p.name) for p in batch]
        paper_lines = "\n".join(
            f"{i + 1}. [{p['year'] or '?'}] {p['author']} — {p['title']}"
            for i, p in enumerate(parsed)
        )
        prompt = ASSIGN_PROMPT.format(collections=col_block, papers=paper_lines)

        msg = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            batch_result = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  ERROR: invalid JSON in batch {b_idx + 1}, skipping.")
            print(raw[:500])
            continue

        for str_idx, col_list in batch_result.items():
            paper_idx = int(str_idx) - 1
            if 0 <= paper_idx < len(batch):
                filename = batch[paper_idx].name
                # Validate collection names
                valid = [c for c in col_list if c in col_names]
                all_assignments[filename] = valid

    # Pretty-print summary
    print(f"\n{'='*65}")
    print(f"PAPER ASSIGNMENTS  ({len(all_assignments)} papers)")
    print(f"{'='*65}")

    col_counts = {c: 0 for c in col_names}
    multi = 0
    unassigned = []

    for fname, cols in sorted(all_assignments.items()):
        for c in cols:
            col_counts[c] += 1
        if len(cols) > 1:
            multi += 1
        if not cols:
            unassigned.append(fname)

    print("\nPapers per collection:")
    for name, count in col_counts.items():
        print(f"  {count:>4}  {name}")
    print(f"\n  {multi} papers assigned to 2+ collections")
    print(f"  {len(unassigned)} papers unassigned")
    if unassigned:
        print("\nUnassigned:")
        for f in unassigned:
            print(f"  · {f}")

    # Save
    out = {
        "collections": col_names,
        "assignments": all_assignments,
        "unassigned": unassigned,
        "excluded": list(exclude_files),
    }
    out_path = here / "paper_assignments.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {out_path}")
    print("Review and edit assignments, then proceed to Phase 2 import.")


if __name__ == "__main__":
    main()
