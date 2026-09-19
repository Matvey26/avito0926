"""Build character and word indexes and run their candidate evaluations."""

from __future__ import annotations

import gc
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from evaluation import evaluate_grid
from retrieval import CandidateRows, retrieve
from settings import CHAR_NGRAMS, NORMALIZATIONS, WORD_NGRAMS
from text_processing import (
    TokenNormalizer,
    build_word_documents,
    build_word_queries,
    normalize_text,
)


CharRetrievals = dict[str, dict[str, CandidateRows]]


def build_char_retrievals(
    items: pd.DataFrame,
    validations: dict[str, pd.DataFrame],
    item_categories: np.ndarray,
) -> tuple[CharRetrievals, list[dict[str, object]]]:
    titles = items["item_title_raw"].fillna("").map(normalize_text)
    retrievals: CharRetrievals = {}
    diagnostics: list[dict[str, object]] = []
    for name, ngram_range in CHAR_NGRAMS.items():
        print(f"Fitting char/{name} index...", flush=True)
        started = time.monotonic()
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            min_df=2,
            max_features=150_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        matrix = vectorizer.fit_transform(titles)
        transposed = matrix.T.tocsr()
        details = {
            "char_ngram": name,
            "ngram_range": list(ngram_range),
            "vocabulary_size": len(vectorizer.vocabulary_),
            "shape": list(matrix.shape),
            "nnz": int(matrix.nnz),
            "indexing_seconds": time.monotonic() - started,
        }
        del matrix
        retrievals[name] = {}
        for mode, validation in validations.items():
            print(f"  Retrieving char/{name}/{mode}...", flush=True)
            queries = vectorizer.transform(
                validation["search_query"].fillna("").map(normalize_text)
            )
            retrievals[name][mode] = retrieve(
                queries, transposed, validation, item_categories
            )
            del queries
        diagnostics.append(details)
        del vectorizer, transposed
        gc.collect()
    del titles
    return retrievals, diagnostics


def evaluate_word_indexes(
    items: pd.DataFrame,
    validations: dict[str, pd.DataFrame],
    relevance_rows: dict[str, dict[tuple[object, ...], frozenset[int]]],
    bootstrap_weights: dict[str, np.ndarray],
    char_retrievals: CharRetrievals,
    item_categories: np.ndarray,
    item_locations: np.ndarray,
    category_fallbacks: dict[int, np.ndarray],
    location_fallbacks: dict[tuple[int, int], np.ndarray],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    result_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for normalization in NORMALIZATIONS:
        print(f"Preprocessing word documents: {normalization}...", flush=True)
        preprocessor = TokenNormalizer(normalization)
        preprocessing_started = time.monotonic()
        documents = build_word_documents(items, preprocessor)
        query_documents = {
            mode: build_word_queries(validation, preprocessor, mode)
            for mode, validation in validations.items()
        }
        preprocessing_seconds = time.monotonic() - preprocessing_started

        for word_name, ngram_range in WORD_NGRAMS.items():
            print(f"Fitting word/{normalization}/{word_name} index...", flush=True)
            indexing_started = time.monotonic()
            vectorizer = TfidfVectorizer(
                ngram_range=ngram_range,
                min_df=2,
                max_df=0.99,
                max_features=200_000,
                sublinear_tf=True,
                dtype=np.float32,
            )
            matrix = vectorizer.fit_transform(documents)
            transposed = matrix.T.tocsr()
            details = {
                "normalization": normalization,
                "word_ngram": word_name,
                "ngram_range": list(ngram_range),
                "vocabulary_size": len(vectorizer.vocabulary_),
                "shape": list(matrix.shape),
                "nnz": int(matrix.nnz),
                "preprocessing_seconds": preprocessing_seconds,
                "indexing_seconds": time.monotonic() - indexing_started,
                "normalization_cache_size": len(preprocessor.cache),
            }
            del matrix

            for mode, validation in validations.items():
                print(
                    f"  Evaluating word/{normalization}/{word_name}/{mode}...",
                    flush=True,
                )
                query_matrix = vectorizer.transform(query_documents[mode])
                word_retrieval = retrieve(
                    query_matrix, transposed, validation, item_categories
                )
                mode_char_retrievals = {
                    name: values[mode] for name, values in char_retrievals.items()
                }
                rows, bootstraps = evaluate_grid(
                    normalization,
                    word_name,
                    mode,
                    validation,
                    relevance_rows[mode],
                    bootstrap_weights[mode],
                    word_retrieval,
                    mode_char_retrievals,
                    item_locations,
                    category_fallbacks,
                    location_fallbacks,
                )
                result_rows.extend(rows)
                bootstrap_rows.extend(bootstraps)
                del query_matrix, word_retrieval
            diagnostics.append(details)
            del vectorizer, transposed
            gc.collect()
        del documents, query_documents, preprocessor
        gc.collect()
    return result_rows, bootstrap_rows, diagnostics
