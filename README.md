# Knowledge Vault

AI-powered tools for building and querying a personal academic library in [Zotero](https://www.zotero.org/). Import hundreds of PDFs with proper metadata, organise them into collections automatically, and generate literature reviews from natural-language queries — all from the command line.

## What it does

| Script | Purpose |
|--------|---------|
| `scripts/zotero_import.py` | Bulk-import a folder of PDFs into Zotero with LLM-extracted metadata and CrossRef enrichment |
| `scripts/analyze_library.py` | Propose a collection taxonomy for a large PDF library, then assign every paper to one or more collections |
| `scripts/import_library.py` | Phase-2 importer: creates Zotero collections and imports papers using the assignments from `analyze_library.py` |
| `scripts/repair_attachments.py` | Fix missing local PDF files after a bulk import |
| `scripts/review.py` | Generate a literature review from a natural-language query: searches Zotero, fetches full texts, synthesises with Claude |

## Requirements

- Python 3.11+
- A [Zotero](https://www.zotero.org/) account with API access
- An [Anthropic API key](https://console.anthropic.com/)

Install dependencies:

```bash
pip install anthropic pyzotero pymupdf requests python-dotenv
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/knowledge-vault.git
cd knowledge-vault
```

2. Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

```env
ANTHROPIC_API_KEY=sk-ant-...
ZOTERO_API_KEY=...
ZOTERO_USER_ID=...
CROSSREF_MAILTO=you@example.com   # optional but polite
```

Your Zotero API key and user ID are available at https://www.zotero.org/settings/keys.

3. (Optional) Add the scripts to your PATH:

```bash
ln -s "$(pwd)/scripts/review.py"              ~/bin/zotero-review
ln -s "$(pwd)/scripts/zotero_import.py"       ~/bin/zotero-import
ln -s "$(pwd)/scripts/analyze_library.py"     ~/bin/zotero-analyze
ln -s "$(pwd)/scripts/import_library.py"      ~/bin/zotero-import-library
ln -s "$(pwd)/scripts/repair_attachments.py"  ~/bin/zotero-repair-attachments
chmod +x scripts/*.py
```

---

## Workflows

### 1. Generate a literature review

Search your Zotero library, fetch full texts, and synthesise a markdown review in one command:

```bash
python scripts/review.py "impact of generative AI on labour markets" --out review.md
```

Options:

```
--limit N        Papers to include in synthesis (default: 12)
--candidates N   Candidates fetched before Claude re-ranks (default: 40)
--out FILE       Output file (default: auto-named from query)
```

Full texts are cached in `text_cache/` so re-running the same query is fast and cheap.

---

### 2. Bulk import a folder of PDFs

For importing a small, self-contained folder of PDFs with automatic metadata extraction:

```bash
python scripts/zotero_import.py /path/to/pdfs/
python scripts/zotero_import.py /path/to/pdfs/ --dry-run   # preview only
```

Each PDF goes through:
1. Text extraction (PyMuPDF)
2. Metadata extraction via Claude (title, authors, year, DOI, type…)
3. CrossRef enrichment
4. Collection classification
5. Upload to Zotero with PDF attachment

---

### 3. Import a large library (two-phase)

For hundreds of PDFs (e.g. a Mendeley export), use the two-phase workflow.

**Phase 1 — propose and assign a taxonomy:**

```bash
# Step 1: Claude proposes 8–15 collections based on your PDF filenames
python scripts/analyze_library.py /path/to/pdfs/

# Review and edit proposed_collections.json, then:

# Step 2: Claude assigns every paper to one or more collections
python scripts/analyze_library.py /path/to/pdfs/ --assign
```

This produces `paper_assignments.json`. Review and edit it before proceeding.

**Phase 2 — import:**

```bash
python scripts/import_library.py /path/to/pdfs/
```

The importer is resumable: progress is saved to `import_progress.json` and interrupted runs can be continued by re-running the same command.

---

### 4. Repair missing PDF attachments

If PDFs were registered in Zotero but are not available locally:

```bash
python scripts/repair_attachments.py /path/to/original/pdfs/
python scripts/repair_attachments.py /path/to/original/pdfs/ --dry-run
```

---

## Notes

- PDFs with no extractable text (scanned images without OCR) will fail metadata extraction. Use an OCR tool beforehand if needed.
- The `review.py` script uses Zotero's full-text search index. Make sure Zotero has indexed your library before running it (Zotero desktop does this automatically in the background).
- CrossRef lookups use a confidence threshold (score ≥ 50) to avoid false matches.

## License

MIT
