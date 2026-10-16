"""Shared fixtures for the unit and integration test suites."""

import os
from collections import Counter
from collections.abc import Callable, Iterator
from uuid import uuid4

import dotenv
import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from classes.raw_faq import RawFaq
from classes.sentence_embedder import SentenceEmbedder

dotenv.load_dotenv()


class StubRawFaq:
    """RawFaq stand-in that serves an already downloaded corpus.

    Integration tests download the corpus once and let every adapter build its
    index from that same snapshot, instead of hitting the remote FAQ once per
    adapter.
    """

    def __init__(self, items: list[dict[str, str]]) -> None:
        self._items = items

    @property
    def items(self) -> list[dict[str, str]]:
        return [dict(item) for item in self._items]


@pytest.fixture(scope="session")
def real_faq_items() -> list[dict[str, str]]:
    """The real FAQ corpus downloaded from the DataTalks.Club site."""
    items = RawFaq().items
    assert items, "The remote FAQ returned no items"
    return items


@pytest.fixture(scope="session")
def raw_faq_factory(
    real_faq_items: list[dict[str, str]],
) -> Callable[[], StubRawFaq]:
    """A zero-argument factory returning a RawFaq over the downloaded corpus."""
    return lambda: StubRawFaq(real_faq_items)


@pytest.fixture(scope="session")
def target_faq_item(real_faq_items: list[dict[str, str]]) -> dict[str, str]:
    """A corpus item with a unique and long question, used as search target."""
    counts = Counter(item["question"] for item in real_faq_items)
    for item in real_faq_items:
        if counts[item["question"]] == 1 and len(item["question"].split()) >= 5:
            return item
    raise AssertionError("The corpus has no unique question with five words")


@pytest.fixture(scope="session")
def sentence_embedder() -> SentenceEmbedder:
    """A single local sentence-transformers embedder shared by the suite."""
    return SentenceEmbedder()


@pytest.fixture(scope="module")
def pg_test_dsn() -> Iterator[str]:
    """A throwaway PostgreSQL database on the server of ``DATABASE_URL``.

    The fixture is module-scoped, so each integration module that requests it
    gets its own fresh database, created before the module runs and dropped
    right after. It skips the tests when ``DATABASE_URL`` is missing or the
    server is unreachable.
    """
    base_dsn = os.getenv("DATABASE_URL")
    if not base_dsn or not base_dsn.strip():
        pytest.skip("DATABASE_URL is not set, cannot test pgvector end to end")

    try:
        admin = psycopg.connect(base_dsn, autocommit=True, connect_timeout=5)
    except psycopg.OperationalError as exc:
        pytest.skip(
            f"PostgreSQL is not reachable ({exc}); run `docker compose up -d db`"
        )

    database = f"faq_test_{uuid4().hex[:8]}"
    with admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
        params = conninfo_to_dict(base_dsn)
        params["dbname"] = database
        yield make_conninfo(**params)
        admin.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                sql.Identifier(database)
            )
        )
