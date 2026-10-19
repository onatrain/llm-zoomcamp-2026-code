"""In-memory keyword index backed by ``minsearch.Index``."""

from functools import partial
from typing import Sequence

from minsearch import Index

from classes.keyword_index import KeywordIndex
from classes.managed_resource import ManagedResource
from utils.normalize import normalize_index_param


def _clear_index(index: Index) -> None:
    """Release the heavy in-memory structures held by a minsearch Index.

    This smell is caused by the lack of explicit memory cleaning in
    minsearch.Index

    Args:
        index: The index whose internal buffers must be cleared.
    """
    index.docs = []
    index.text_matrices = {}
    index.keyword_df = None
    index.numeric_df = None
    index.date_df = None
    index._filter = None
    index.vectorizers = {}


class InMemoryIndex(ManagedResource[Index], KeywordIndex):
    """In-memory keyword index using minsearch.

    The index is volatile: its state is never persisted, so opening it always
    yields a fresh, empty instance. ``reset()`` is therefore a no-op that
    exists only to satisfy the ``KeywordIndex`` contract invoked by
    ``LexicalFAQ.build_index``.

    The underlying ``minsearch.Index`` has no ``close()`` method, so a closer
    that clears its heavy buffers is registered with ``ManagedResource`` and
    runs both on explicit ``close()`` and on garbage collection.
    """

    closed_message = "Index closed. You must open() it first."

    def __init__(self) -> None:
        """Initialize a closed, empty in-memory index."""
        super().__init__()

    def open(self) -> None:
        """Create a fresh, empty in-memory index.

        If the index is already open, it is closed first.
        """
        if self.is_open:
            self.close()

        index = Index(
            text_fields=["question", "section", "answer"],
            keyword_fields=["course"],
        )
        self._register_resource(index, partial(_clear_index, index))

    def reset(self) -> None:
        """No-op reset for the volatile in-memory index.

        The index is never persisted and ``open()`` already discards any
        previous content by creating a new empty ``Index``, so there is nothing
        to clear. The method exists only to honour the ``KeywordIndex``
        contract invoked by ``LexicalFAQ.build_index``.
        """
        # Volatile index: ``open()`` already discards any previous content by
        # creating a brand-new, empty ``Index``. There is nothing to clear here;
        # the method only honours the KeywordIndex/LexicalFAQ contract.
        pass

    def load(self, documents: Sequence[dict[str, str]]) -> None:
        """Replace the index content with the given documents.

        Args:
            documents: Documents to fit into the index.
        """
        if not self.is_open:
            self.open()
        index = self._require_resource()
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
        return len(self._require_resource().docs)

    def close(self) -> None:
        """Close the index and release managed resources."""
        resource = self._resource
        if resource is not None:
            _clear_index(resource)
        self._release_resource()
