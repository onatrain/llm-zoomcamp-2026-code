"""In-memory keyword index backed by ``minsearch.Index``."""

from typing import Sequence

from minsearch import Index

from classes.keyword_index import KeywordIndex
from classes.managed_resource import ManagedResource


class InMemoryIndex(ManagedResource[Index], KeywordIndex):
    """In-memory keyword index using minsearch.

    The index is volatile: its state is never persisted, so opening it always
    yields a fresh, empty instance. ``reset()`` is therefore a no-op that
    exists only to satisfy the ``KeywordIndex`` contract invoked by
    ``LexicalFAQ.build_index``.
    """

    closed_message = "Index closed. You must open() it first."

    def __init__(self) -> None:
        """Initialize a closed, empty in-memory index."""
        super().__init__()
        self._index: Index | None = None

    @property
    def is_open(self) -> bool:
        """Whether the index is currently open."""
        return self._index is not None

    def open(self) -> None:
        """Create a fresh, empty in-memory index.

        If the index is already open, it is closed first.
        """
        if self.is_open:
            self.close()

        self._index = Index(
            text_fields=["question", "section", "answer"],
            keyword_fields=["course"],
        )

    def reset(self) -> None:
        """No-op reset for the volatile in-memory index.

        The index is never persisted and ``open()`` already discards any
        previous content by creating a new empty ``Index``, so there is nothing
        to clear. The method exists only to honour the ``KeywordIndex``
        contract invoked by ``LexicalFAQ.build_index``.
        """
        pass

    def load(self, documents: Sequence[dict[str, str]]) -> None:
        """Replace the index content with the given documents.

        Args:
            documents: Documents to fit into the index.
        """
        if self._index is None:
            self.open()
        index = self._index
        if index is None:
            raise RuntimeError(self.closed_message)
        index.fit(list(documents))

    def search(
        self,
        question: str,
        filter_dict: dict[str, str] | None = None,
        boost_dict: dict[str, float] | None = None,
        num_results: int = 10,
    ) -> list[dict[str, str]]:
        """Search the index.

        Args:
            question: Query text.
            filter_dict: Optional field filters.
            boost_dict: Optional field boosts.
            num_results: Maximum number of results.

        Returns:
            Matching documents.
        """
        index = self._index
        if index is None:
            raise RuntimeError(self.closed_message)

        results = index.search(
            question,
            filter_dict=filter_dict,
            boost_dict=boost_dict,
            num_results=num_results,
        )
        return list(results)

    def size(self) -> int:
        """Return the number of stored documents.

        Returns:
            Document count.
        """
        index = self._index
        if index is None:
            raise RuntimeError(self.closed_message)
        return len(index.docs)

    def close(self) -> None:
        """Close the index and release managed resources."""
        self._index = None
        self._release_resource()
