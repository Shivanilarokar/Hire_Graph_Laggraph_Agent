"""Tests for local document loading."""

from __future__ import annotations

import os

from hiregraph.services import read_document


def test_read_document_accepts_accidentally_wrapped_cli_path(tmp_path):
    document = tmp_path / "job description.md"
    document.write_text("Senior backend engineer", encoding="utf-8")
    wrapped_path = f"{tmp_path}{os.sep}\n  {document.name}"

    assert read_document(wrapped_path) == "Senior backend engineer"
