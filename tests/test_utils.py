"""
Unit tests for scripts/utils.py.

Run with:
    pytest tests/
"""

import sys
from pathlib import Path

# Make scripts/ importable without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest
from utils import (
    ITEM_TYPE_MAP,
    strip_json_fences,
    build_zotero_item,
)


# ── strip_json_fences ─────────────────────────────────────────────────────────

class TestStripJsonFences:
    def test_plain_json_unchanged(self):
        raw = '{"key": "value"}'
        assert strip_json_fences(raw) == raw

    def test_removes_backtick_fence(self):
        raw = "```\n{\"key\": \"value\"}\n```"
        assert strip_json_fences(raw) == '{"key": "value"}'

    def test_removes_json_labelled_fence(self):
        raw = "```json\n{\"key\": \"value\"}\n```"
        assert strip_json_fences(raw) == '{"key": "value"}'

    def test_strips_whitespace(self):
        raw = "   \n{}\n   "
        assert strip_json_fences(raw) == "{}"


# ── ITEM_TYPE_MAP ─────────────────────────────────────────────────────────────

class TestItemTypeMap:
    def test_journal_article(self):
        assert ITEM_TYPE_MAP["journal article"] == "journalArticle"

    def test_book_chapter_aliases(self):
        assert ITEM_TYPE_MAP["book chapter"] == "bookSection"
        assert ITEM_TYPE_MAP["book section"] == "bookSection"

    def test_working_paper_maps_to_report(self):
        assert ITEM_TYPE_MAP["working paper"] == "report"

    def test_other_maps_to_document(self):
        assert ITEM_TYPE_MAP["other"] == "document"

    def test_all_values_are_valid_zotero_types(self):
        valid = {
            "journalArticle", "book", "bookSection", "report",
            "thesis", "conferencePaper", "preprint", "document",
        }
        assert set(ITEM_TYPE_MAP.values()) <= valid


# ── build_zotero_item ─────────────────────────────────────────────────────────

class TestBuildZoteroItem:
    def _base_meta(self, item_type="journal article", **kwargs):
        meta = {
            "title":     "Test Paper",
            "item_type": item_type,
            "authors":   [{"first": "Jane", "last": "Doe"}],
            "editors":   [],
            "year":      "2023",
            "abstract":  "An abstract.",
            "language":  "en",
        }
        meta.update(kwargs)
        return meta

    def test_journal_article_fields(self):
        meta = self._base_meta(
            journal="Nature", volume="10", issue="2",
            pages="100-110", doi="10.1234/test", issn="1234-5678",
        )
        item = build_zotero_item(meta, ["COL1"])
        assert item["itemType"] == "journalArticle"
        assert item["publicationTitle"] == "Nature"
        assert item["DOI"] == "10.1234/test"
        assert item["collections"] == ["COL1"]

    def test_book_fields(self):
        meta = self._base_meta(
            item_type="book",
            publisher="Oxford UP", place="Oxford",
            isbn="978-0-19-000000-0",
        )
        item = build_zotero_item(meta, [])
        assert item["itemType"] == "book"
        assert item["publisher"] == "Oxford UP"
        assert item["ISBN"] == "978-0-19-000000-0"

    def test_book_section_fields(self):
        meta = self._base_meta(
            item_type="book chapter",
            book_title="Handbook of Economics",
            publisher="Elsevier",
        )
        item = build_zotero_item(meta, [])
        assert item["itemType"] == "bookSection"
        assert item["bookTitle"] == "Handbook of Economics"

    def test_report_fields(self):
        meta = self._base_meta(
            item_type="report",
            institution="World Bank",
            place="Washington DC",
        )
        item = build_zotero_item(meta, [])
        assert item["itemType"] == "report"
        assert item["institution"] == "World Bank"

    def test_report_falls_back_to_publisher_for_institution(self):
        meta = self._base_meta(
            item_type="report",
            publisher="IMF",
        )
        item = build_zotero_item(meta, [])
        assert item["institution"] == "IMF"

    def test_unknown_type_becomes_document(self):
        meta = self._base_meta(item_type="other")
        item = build_zotero_item(meta, [])
        assert item["itemType"] == "document"

    def test_creators_include_authors_and_editors(self):
        meta = self._base_meta()
        meta["editors"] = [{"first": "Ed", "last": "Itor"}]
        item = build_zotero_item(meta, [])
        creator_types = [c["creatorType"] for c in item["creators"]]
        assert "author" in creator_types
        assert "editor" in creator_types

    def test_url_fallback(self):
        meta = self._base_meta(url="https://example.com/paper")
        item = build_zotero_item(meta, [])
        assert item.get("url") == "https://example.com/paper"

    def test_multiple_collections(self):
        meta = self._base_meta()
        item = build_zotero_item(meta, ["A", "B", "C"])
        assert item["collections"] == ["A", "B", "C"]

    def test_none_fields_become_empty_strings(self):
        meta = self._base_meta(journal=None, volume=None, issue=None)
        item = build_zotero_item(meta, [])
        assert item.get("publicationTitle") == ""
        assert item.get("volume") == ""
        assert item.get("issue") == ""
