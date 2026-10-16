"""Unit tests for the SQLite-backed keyword index."""

from pathlib import Path

import pytest

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


@pytest.fixture
def index(tmp_path: Path) -> LocalDbIndex:
    return LocalDbIndex(str(tmp_path / "faq.db"))


def test_add_all_indexes_documents(index: LocalDbIndex) -> None:
    index.add_all(DOCUMENTS)

    assert index.size() == len(DOCUMENTS)


def test_search_returns_matching_documents(index: LocalDbIndex) -> None:
    index.add_all(DOCUMENTS)

    results = index.search("virtual environment", num_results=5)

    assert results
    assert any("virtual environment" in result["answer"].lower() for result in results)


def test_search_returns_the_stored_fields(index: LocalDbIndex) -> None:
    index.add_all(DOCUMENTS)

    results = index.search("prerequisites", num_results=5)

    assert results
    assert set(results[0]) >= {"course", "section", "question", "answer"}


def test_search_respects_the_course_filter(index: LocalDbIndex) -> None:
    index.add_all(DOCUMENTS)

    results = index.search(
        "prerequisites", filter_dict={"course": "mlops-zoomcamp"}, num_results=5
    )

    assert results
    assert all(result["course"] == "mlops-zoomcamp" for result in results)


def test_reset_clears_the_index(index: LocalDbIndex) -> None:
    index.add_all(DOCUMENTS)

    index.reset()

    assert index.size() == 0


def test_add_reopens_a_closed_index(index: LocalDbIndex) -> None:
    index.close()

    index.add(DOCUMENTS[0])

    assert index.is_open
    assert index.size() == 1


def test_search_requires_an_open_index(index: LocalDbIndex) -> None:
    with pytest.raises(RuntimeError, match="closed"):
        index.search("anything")


def test_size_requires_an_open_index(index: LocalDbIndex) -> None:
    with pytest.raises(RuntimeError, match="closed"):
        index.size()


def test_documents_persist_across_instances(tmp_path: Path) -> None:
    db_path = str(tmp_path / "faq.db")
    first = LocalDbIndex(db_path)
    first.add_all(DOCUMENTS)
    first.close()

    second = LocalDbIndex(db_path)
    second.open()

    assert second.size() == len(DOCUMENTS)
    assert second.search("prerequisites", num_results=5)
    second.close()
