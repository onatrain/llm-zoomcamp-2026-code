"""Unit tests for the SQLite-backed vector index."""

from pathlib import Path

import numpy as np
import pytest

from classes.local_vector_db_index import LocalVectorDbIndex

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

VECTORS = [np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 1.0])]
QUERY = np.array([1.0, 0.5, 0.0])


@pytest.fixture
def index(tmp_path: Path) -> LocalVectorDbIndex:
    return LocalVectorDbIndex(str(tmp_path / "vector_faq.db"))


def test_load_indexes_documents(index: LocalVectorDbIndex) -> None:
    index.load(DOCUMENTS, VECTORS)

    assert index.size() == len(DOCUMENTS)


def test_search_returns_the_closest_documents_first(
    index: LocalVectorDbIndex,
) -> None:
    index.load(DOCUMENTS, VECTORS)

    results = index.search(QUERY, num_results=2)

    assert len(results) == 2
    assert results[0]["question"] == DOCUMENTS[0]["question"]


def test_search_respects_the_course_filter(index: LocalVectorDbIndex) -> None:
    index.load(DOCUMENTS, VECTORS)

    results = index.search(QUERY, filter_dict={"course": "mlops-zoomcamp"}, num_results=5)

    assert results
    assert all(result["course"] == "mlops-zoomcamp" for result in results)


def test_load_rejects_mismatched_documents_and_vectors(
    index: LocalVectorDbIndex,
) -> None:
    with pytest.raises(ValueError, match="length"):
        index.load(DOCUMENTS, VECTORS[:1])


def test_load_indexes_a_single_document(index: LocalVectorDbIndex) -> None:
    index.load([DOCUMENTS[0]], [VECTORS[0]])

    assert index.size() == 1


def test_reset_clears_the_index(index: LocalVectorDbIndex) -> None:
    index.load(DOCUMENTS, VECTORS)

    index.reset()

    assert index.size() == 0


def test_search_requires_an_open_index(index: LocalVectorDbIndex) -> None:
    with pytest.raises(RuntimeError, match="closed"):
        index.search(QUERY)


def test_documents_persist_across_instances(tmp_path: Path) -> None:
    db_path = str(tmp_path / "vector_faq.db")
    first = LocalVectorDbIndex(db_path)
    first.load(DOCUMENTS, VECTORS)
    first.close()

    second = LocalVectorDbIndex(db_path)
    second.open()

    assert second.size() == len(DOCUMENTS)
    results = second.search(QUERY, num_results=2)
    assert results[0]["question"] == DOCUMENTS[0]["question"]
    second.close()
