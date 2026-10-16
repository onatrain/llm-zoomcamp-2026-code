"""End-to-end tests of LexicalFAQ over the SQLite-backed keyword index.

These tests use the real FAQ corpus and build a real SQLite index. The corpus
is downloaded once per session and reused by every adapter through a
``RawFaq`` stand-in, so the remote site is hit only when fetching it.
"""

from collections.abc import Callable, Iterator

import pytest

from classes import lexical_faq
from classes.lexical_faq import LexicalFAQ
from classes.local_db_index import LocalDbIndex


@pytest.fixture(scope="module")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    return str(tmp_path_factory.mktemp("local_db") / "faq.db")


@pytest.fixture(scope="module")
def faq(db_path: str, raw_faq_factory: Callable[[], object]) -> Iterator[LexicalFAQ]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(lexical_faq, "RawFaq", raw_faq_factory)
        faq = LexicalFAQ(LocalDbIndex(db_path))
        faq.build_index()

    yield faq
    faq.close()


def test_build_index_stores_the_whole_real_corpus(
    faq: LexicalFAQ, real_faq_items: list[dict[str, str]]
) -> None:
    assert faq.count == len(real_faq_items)


def test_search_finds_the_target_document(
    faq: LexicalFAQ, target_faq_item: dict[str, str]
) -> None:
    results = faq.search(target_faq_item["question"], num_results=10)

    assert any(result["question"] == target_faq_item["question"] for result in results)


def test_search_respects_the_course_filter(
    faq: LexicalFAQ, target_faq_item: dict[str, str]
) -> None:
    results = faq.search(
        target_faq_item["question"],
        filter_dict={"course": target_faq_item["course"]},
        num_results=10,
    )

    assert results
    assert all(result["course"] == target_faq_item["course"] for result in results)


def test_index_persists_across_instances(
    db_path: str, real_faq_items: list[dict[str, str]], target_faq_item: dict[str, str]
) -> None:
    reopened = LexicalFAQ(LocalDbIndex(db_path))
    reopened.open()

    assert reopened.count == len(real_faq_items)
    results = reopened.search(target_faq_item["question"], num_results=10)
    assert any(result["question"] == target_faq_item["question"] for result in results)

    reopened.close()
