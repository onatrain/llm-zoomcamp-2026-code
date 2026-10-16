"""Unit tests for the LexicalFAQ adapter with a stubbed RawFaq source."""

from pathlib import Path

import pytest

from classes import lexical_faq
from classes.lexical_faq import LexicalFAQ
from classes.local_db_index import LocalDbIndex

DOCUMENTS = [
    {
        "course": "llm-zoomcamp",
        "section": "General",
        "question": "How do I install the course environment?",
        "answer": "Create a virtual environment and install the requirements.",
    },
    {
        "course": "mlops-zoomcamp",
        "section": "General",
        "question": "What are the prerequisites for the course?",
        "answer": "Knowledge of Python, Docker and machine learning basics.",
    },
]


class StubRawFaq:
    """RawFaq replacement that serves a fixed document list."""

    @property
    def items(self) -> list[dict[str, str]]:
        return [dict(document) for document in DOCUMENTS]


@pytest.fixture
def faq(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LexicalFAQ:
    monkeypatch.setattr(lexical_faq, "RawFaq", StubRawFaq)
    return LexicalFAQ(LocalDbIndex(str(tmp_path / "faq.db")))


def test_build_index_stores_the_raw_documents(faq: LexicalFAQ) -> None:
    faq.build_index()

    assert faq.count == len(DOCUMENTS)


def test_build_index_replaces_the_previous_content(faq: LexicalFAQ) -> None:
    faq.build_index()

    faq.build_index()

    assert faq.count == len(DOCUMENTS)


def test_search_finds_a_built_document(faq: LexicalFAQ) -> None:
    faq.build_index()

    results = faq.search("prerequisites", num_results=5)

    assert any(result["question"] == DOCUMENTS[1]["question"] for result in results)


def test_count_requires_an_open_index(tmp_path: Path) -> None:
    faq = LexicalFAQ(LocalDbIndex(str(tmp_path / "faq.db")))

    with pytest.raises(RuntimeError, match="closed"):
        _ = faq.count
