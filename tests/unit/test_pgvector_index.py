"""Unit tests for the pgvector index that never touch a database."""

import numpy as np
import pytest

from classes.pgvector_index import PgvectorIndex, vector_literal

DSN = "postgresql://postgres:postgres@localhost:5432/faq"

DOCUMENT = {
    "course": "llm-zoomcamp",
    "section": "General",
    "question": "How do I install the environment?",
    "answer": "Create a virtual environment and install the requirements.",
}


def test_vector_literal_formats_the_pgvector_text_representation() -> None:
    assert vector_literal(np.array([1.0, -2.5, 0.0])) == "[1.0,-2.5,0.0]"


def test_empty_dsn_is_rejected() -> None:
    with pytest.raises(ValueError, match="DSN"):
        PgvectorIndex("   ", dimension=3)


@pytest.mark.parametrize("dimension", [0, -1])
def test_non_positive_dimension_is_rejected(dimension: int) -> None:
    with pytest.raises(ValueError, match="Dimension"):
        PgvectorIndex(DSN, dimension=dimension)


def test_load_rejects_mismatched_documents_and_vectors() -> None:
    index = PgvectorIndex(DSN, dimension=3)
    with pytest.raises(ValueError, match="vectors"):
        index.load([DOCUMENT], [])


def test_size_requires_an_open_connection() -> None:
    index = PgvectorIndex(DSN, dimension=3)
    with pytest.raises(RuntimeError, match="closed"):
        index.size()
