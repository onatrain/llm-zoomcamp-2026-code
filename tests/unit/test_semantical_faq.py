"""Unit tests for the SemanticalFAQ adapter with a stubbed source and embedder."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from classes import semantical_faq
from classes.embedder import Embedder
from classes.local_vector_db_index import LocalVectorDbIndex
from classes.pgvector_index import PgvectorIndex
from classes.semantical_faq import SemanticalFAQ

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

QUERY = np.array([1.0, 0.5, 0.0])


class StubRawFaq:
    """RawFaq replacement that serves a fixed document list."""

    @property
    def items(self) -> list[dict[str, str]]:
        return [dict(document) for document in DOCUMENTS]


class StubEmbedder(Embedder):
    """Deterministic embedder that records the embedded texts."""

    def __init__(self) -> None:
        self.embedded_texts: list[str] = []

    @property
    def embedding_dimension(self) -> int:
        return 3

    def embed_texts(self, texts: list[str]) -> NDArray[np.float64]:
        self.embedded_texts = list(texts)
        return np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]][: len(texts)])

    def encode_text(self, text: str) -> NDArray[np.float64]:
        return QUERY


@pytest.fixture
def embedder() -> StubEmbedder:
    return StubEmbedder()


@pytest.fixture
def faq(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedder: StubEmbedder
) -> SemanticalFAQ:
    monkeypatch.setattr(semantical_faq, "RawFaq", StubRawFaq)
    return SemanticalFAQ(LocalVectorDbIndex(str(tmp_path / "vector_faq.db")), embedder)


def test_build_index_stores_the_raw_documents(faq: SemanticalFAQ) -> None:
    faq.build_index()

    assert faq.count == len(DOCUMENTS)


def test_build_index_embeds_question_and_answer(
    faq: SemanticalFAQ, embedder: StubEmbedder
) -> None:
    faq.build_index()

    assert embedder.embedded_texts == [
        " ".join((document["question"], document["answer"])) for document in DOCUMENTS
    ]


def test_search_returns_the_closest_document(faq: SemanticalFAQ) -> None:
    faq.build_index()

    results = faq.search("how do I install the environment?", num_results=2)

    assert results
    assert results[0]["question"] == DOCUMENTS[0]["question"]


def test_search_respects_the_course_filter(faq: SemanticalFAQ) -> None:
    faq.build_index()

    results = faq.search(
        "any question", filter_dict={"course": "mlops-zoomcamp"}, num_results=5
    )

    assert results
    assert all(result["course"] == "mlops-zoomcamp" for result in results)


def test_open_rejects_an_index_dimension_mismatch(
    embedder: StubEmbedder,
) -> None:
    faq = SemanticalFAQ(
        PgvectorIndex("postgresql://localhost/faq", dimension=7), embedder
    )

    with pytest.raises(ValueError, match="dimension"):
        faq.open()


def test_count_requires_an_open_index(tmp_path: Path) -> None:
    faq = SemanticalFAQ(LocalVectorDbIndex(str(tmp_path / "faq.db")), StubEmbedder())

    with pytest.raises(RuntimeError, match="closed"):
        _ = faq.count
