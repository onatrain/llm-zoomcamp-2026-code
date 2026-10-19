"""SQLite-backed keyword index using ``sqlitesearch.TextSearchIndex``."""

from typing import Sequence

from sqlitesearch import TextSearchIndex

from classes.keyword_index import KeywordIndex
from classes.managed_resource import ManagedResource
from utils.normalize import normalize_index_param


class LocalDbIndex(ManagedResource[TextSearchIndex], KeywordIndex):
    """Keyword index persisted in SQLite via sqlitesearch.

    The store is persistent but treated as read-only and immutable: its content
    is replaced wholesale through ``reset()`` followed by ``load()`` rather than
    appended to incrementally.
    """

    closed_message = "Index closed. You must open() it first."

    def __init__(self, db_path: str = "./faq.db") -> None:
        """Initialize a closed index backed by the given SQLite file.

        Args:
            db_path: Path to the SQLite database file.
        """
        super().__init__()
        self._db_path = db_path

    def open(self) -> None:
        """Open (or reopen) the connection to the SQLite store.

        If the index is already open, it is closed first.
        """
        if self.is_open:
            self.close()

        index = TextSearchIndex(
            text_fields=["question", "section", "answer"],
            keyword_fields=["course"],
            db_path=self._db_path,
        )
        self._register_resource(index, index.close)

    def reset(self) -> None:
        """Destroy all stored content so the index can be rebuilt.

        The index must already be open; this method never opens a connection
        itself. The caller is responsible for opening (and closing) the index.

        Raises:
            RuntimeError: If the index is closed.
        """
        index = self._require_resource()
        index.clear()

    def load(self, documents: Sequence[dict[str, str]]) -> None:
        """Load documents into the index.

        Opens the index if it is closed. Assumes the index is empty; otherwise
        ``TextSearchIndex.fit`` raises ``ValueError`` because it refuses to
        append to a non-empty index. Call ``reset()`` first to clear it.

        Args:
            documents: Documents to fit into the index.
        """
        if not self.is_open:
            self.open()
        index = self._require_resource()
        if not documents:
            return
        index.fit(list(documents))

    def search(
        self,
        question: str,
        filter_dict: dict[str, str] | None = None,
        boost_dict: dict[str, float] | None = None,
        num_results: int = 5,
    ) -> list[dict[str, str]]:
        """Search the index.

        Args:
            question: Query text.
            filter_dict: Optional field filters.
            boost_dict: Optional field boosts.
            num_results: Maximum number of results.

        Returns:
            Matching documents.

        Raises:
            RuntimeError: If the index is closed.
        """
        index = self._require_resource()
        normalized_filter = normalize_index_param(filter_dict)
        normalized_boost = normalize_index_param(boost_dict)
        results = index.search(
            question,
            filter_dict=normalized_filter,
            boost_dict=normalized_boost,
            num_results=num_results,
        )
        return list(results)

    def size(self) -> int:
        """Return the number of stored documents.

        Returns:
            Document count.

        Raises:
            RuntimeError: If the index is closed.
        """
        index = self._require_resource()
        return int(index.count())

    def close(self) -> None:
        """Close the index and release managed resources."""
        resource = self._resource
        if resource is not None:
            resource.close()
        self._release_resource()
