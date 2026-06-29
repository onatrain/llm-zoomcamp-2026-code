"""Type stubs for minsearch.

Minimalistic text and vector search engines that use Scikit-Learn and Pandas.

This package provides type information for the minsearch library.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Document = Dict[str, object]
FilterDict = Dict[str, Union[str, List[Tuple[str, object]]]]
BoostDict = Dict[str, Union[int, float]]
StemmerOption = Optional[Union[Literal["porter", "snowball", "lancaster"], Callable[[str], str]]]
StopWordsOption = Union[Literal["english"], Set[str], None]
HighlightFormat = Union[str, Tuple[str, str], Callable[[str], str]]

# ---------------------------------------------------------------------------
# Stemmers
# ---------------------------------------------------------------------------

def get_stemmer(name: StemmerOption) -> Optional[Callable[[str], str]]: ...

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class Tokenizer:
    pattern: re.Pattern[str]
    min_token_length: int
    stop_words: Set[str]
    stemmer: Optional[Callable[[str], str]]

    def __init__(
        self,
        pattern: str = r"[\s\W\d]+",
        stop_words: StopWordsOption = None,
        stemmer: StemmerOption = None,
        min_token_length: int = 2,
    ) -> None: ...
    def tokenize(self, text: str) -> List[str]: ...

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

class FieldData:
    fields: List[str]
    data: Union[pd.DataFrame, Dict[str, List]]

    def __init__(
        self,
        fields: List[str],
        data: Union[pd.DataFrame, Dict[str, List]],
    ) -> None: ...

class Filter:
    def __init__(
        self,
        keyword: FieldData,
        numeric: FieldData,
        date: FieldData,
        num_docs: int,
    ) -> None: ...
    def apply(self, filter_dict: FilterDict) -> NDArray[np.float64]: ...
    def refresh(
        self,
        keyword_data: Union[pd.DataFrame, Dict[str, List]],
        numeric_data: Union[pd.DataFrame, Dict[str, List]],
        date_data: Union[pd.DataFrame, Dict[str, List]],
        num_docs: int,
    ) -> None: ...

# ---------------------------------------------------------------------------
# Highlighter
# ---------------------------------------------------------------------------

class Highlighter:
    highlight_fields: List[str]
    skip_fields: Set[str]
    max_matches: int
    snippet_size: int
    highlight_format: HighlightFormat
    tokenizer: Tokenizer

    def __init__(
        self,
        highlight_fields: List[str],
        skip_fields: Optional[List[str]] = None,
        max_matches: int = 5,
        snippet_size: int = 200,
        highlight_format: HighlightFormat = "**",
        tokenizer: Optional[Tokenizer] = None,
    ) -> None: ...
    def highlight(self, query: str, results: List[Document]) -> List[Document]: ...

# ---------------------------------------------------------------------------
# Index (TF-IDF based)
# ---------------------------------------------------------------------------

class Index:
    text_fields: List[str]
    keyword_fields: List[str]
    numeric_fields: List[str]
    date_fields: List[str]
    vectorizers: Dict[str, TfidfVectorizer]
    keyword_df: Optional[pd.DataFrame]
    numeric_df: Optional[pd.DataFrame]
    date_df: Optional[pd.DataFrame]
    text_matrices: Dict[str, object]
    docs: List[Document]
    _filter: Optional[Filter]

    def __init__(
        self,
        text_fields: List[str],
        keyword_fields: Optional[List[str]] = None,
        numeric_fields: Optional[List[str]] = None,
        date_fields: Optional[List[str]] = None,
        vectorizer_params: Optional[Dict[str, object]] = None,
    ) -> None: ...
    def fit(self, docs: List[Document]) -> Index: ...
    def search(
        self,
        query: str,
        filter_dict: Optional[FilterDict] = None,
        boost_dict: Optional[BoostDict] = None,
        num_results: int = 10,
        output_ids: bool = False,
    ) -> List[Document]: ...
    def save(self, path: Union[str, Path]) -> None: ...
    @classmethod
    def load(cls, path: Union[str, Path]) -> Index: ...

# ---------------------------------------------------------------------------
# AppendableIndex (inverted index based)
# ---------------------------------------------------------------------------

class AppendableIndex:
    text_fields: List[str]
    keyword_fields: List[str]
    numeric_fields: List[str]
    date_fields: List[str]
    docs: List[Document]
    total_docs: int
    inverted_index: Dict[str, Dict[str, Set[int]]]
    doc_frequencies: Dict[str, Dict[str, int]]
    idf: Dict[str, Dict[str, float]]
    doc_tokens: Dict[str, Dict[int, List[str]]]
    doc_token_counts: Dict[str, Dict[int, Dict[str, int]]]
    doc_norms: Dict[str, Dict[int, float]]
    keyword_data: Dict[str, List]
    numeric_data: Dict[str, List]
    date_data: Dict[str, List]
    vocabularies: Dict[str, Set[str]]
    tokenizer: Tokenizer
    _filter: Filter

    def __init__(
        self,
        text_fields: List[str],
        keyword_fields: Optional[List[str]] = None,
        numeric_fields: Optional[List[str]] = None,
        date_fields: Optional[List[str]] = None,
        tokenizer: Optional[Tokenizer] = None,
    ) -> None: ...
    def fit(self, docs: List[Document]) -> AppendableIndex: ...
    def append(self, doc: Document) -> AppendableIndex: ...
    def search(
        self,
        query: str,
        filter_dict: Optional[FilterDict] = None,
        boost_dict: Optional[BoostDict] = None,
        num_results: int = 10,
        output_ids: bool = False,
    ) -> List[Document]: ...
    def save(self, path: Union[str, Path]) -> None: ...
    @classmethod
    def load(cls, path: Union[str, Path]) -> AppendableIndex: ...

# ---------------------------------------------------------------------------
# VectorSearch (vector similarity based)
# ---------------------------------------------------------------------------

class VectorSearch:
    keyword_fields: List[str]
    numeric_fields: List[str]
    date_fields: List[str]
    vectors: Optional[NDArray[np.float64]]
    keyword_df: Optional[pd.DataFrame]
    numeric_df: Optional[pd.DataFrame]
    date_df: Optional[pd.DataFrame]
    docs: List[Document]
    _filter: Filter

    def __init__(
        self,
        keyword_fields: Optional[List[str]] = None,
        numeric_fields: Optional[List[str]] = None,
        date_fields: Optional[List[str]] = None,
    ) -> None: ...
    def fit(
        self,
        vectors: NDArray[np.float64],
        payload: List[Document],
    ) -> VectorSearch: ...
    def append(
        self,
        vector: NDArray[np.float64],
        doc: Document,
    ) -> VectorSearch: ...
    def append_batch(
        self,
        vectors: NDArray[np.float64],
        payload: List[Document],
    ) -> VectorSearch: ...
    def search(
        self,
        query_vector: NDArray[np.float64],
        filter_dict: Optional[FilterDict] = None,
        num_results: int = 10,
        output_ids: bool = False,
    ) -> List[Document]: ...
    def save(self, path: Union[str, Path]) -> None: ...
    @classmethod
    def load(cls, path: Union[str, Path]) -> VectorSearch: ...
