"""End-to-end tests of SemanticalFAQ over the pgvector store.

These tests require the PostgreSQL service and the real FAQ corpus. A
throwaway database is created on the same server for every test run, so the
database in ``DATABASE_URL`` is never modified. The corpus is downloaded once
per session and reused by every adapter through a ``RawFaq`` stand-in, so the
remote site is hit only when fetching it.
"""

from collections.abc import Callable, Iterator

import pytest

from classes import semantical_faq
from classes.pgvector_index import PgvectorIndex
from classes.semantical_faq import SemanticalFAQ
from classes.sentence_embedder import SentenceEmbedder


@pytest.fixture(scope="module")
def faq(
    pg_test_dsn: str,
    sentence_embedder: SentenceEmbedder,
    raw_faq_factory: Callable[[], object],
) -> Iterator[SemanticalFAQ]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(semantical_faq, "RawFaq", raw_faq_factory)
        faq = SemanticalFAQ(
            PgvectorIndex(pg_test_dsn, dimension=sentence_embedder.embedding_dimension),
            sentence_embedder,
        )
        faq.build_index()

    yield faq
    faq.close()


def test_build_index_stores_the_whole_real_corpus(
    faq: SemanticalFAQ, real_faq_items: list[dict[str, str]]
) -> None:
    assert faq.count == len(real_faq_items)


def test_search_finds_the_target_document(
    faq: SemanticalFAQ, target_faq_item: dict[str, str]
) -> None:
    results = faq.search(target_faq_item["question"], num_results=5)

    assert any(result["question"] == target_faq_item["question"] for result in results)


def test_search_finds_a_paraphrased_question(faq: SemanticalFAQ) -> None:
    results = faq.search("Do I have to pay for the tools used in the course?")

    assert any(
        "free" in result["answer"].lower() or "pay" in result["answer"].lower()
        for result in results
    )


def test_search_respects_the_course_filter(
    faq: SemanticalFAQ, target_faq_item: dict[str, str]
) -> None:
    results = faq.search(
        target_faq_item["question"],
        filter_dict={"course": target_faq_item["course"]},
        num_results=5,
    )

    assert results
    assert all(result["course"] == target_faq_item["course"] for result in results)


def test_index_persists_across_instances(
    pg_test_dsn: str,
    sentence_embedder: SentenceEmbedder,
    real_faq_items: list[dict[str, str]],
    target_faq_item: dict[str, str],
) -> None:
    reopened = SemanticalFAQ(
        PgvectorIndex(pg_test_dsn, dimension=sentence_embedder.embedding_dimension),
        sentence_embedder,
    )
    reopened.open()

    assert reopened.count == len(real_faq_items)
    results = reopened.search(target_faq_item["question"], num_results=5)
    assert any(result["question"] == target_faq_item["question"] for result in results)

    reopened.close()
