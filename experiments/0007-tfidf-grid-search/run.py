#!/usr/bin/env python3
"""Grid-search word/char TF-IDF, morphology, blending, and location boost."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pymorphy3
import snowballstemmer
from sklearn.feature_extraction.text import TfidfVectorizer


SEED = "20260918"
VALIDATION_SIZE = 2_500
TOP_K = 50
CANDIDATE_POOL = 500
CHAR_WEIGHTS = [round(step / 10, 1) for step in range(11)]
LOCATION_BOOSTS = sorted(
    {round(step / 10, 1) for step in range(11)} | {0.56}
)
WORD_NGRAMS = {
    "unigram": (1, 1),
    "unigram_bigram": (1, 2),
    "unigram_to_trigram": (1, 3),
}
CHAR_NGRAMS = {
    f"{minimum}_to_{maximum}": (minimum, maximum)
    for minimum in range(2, 7)
    for maximum in range(minimum, 7)
}
NORMALIZATIONS = ("plain", "russian_snowball", "russian_lemma")
TOKEN_RE = re.compile(r"(?u)\b\w{2,}\b")
RUSSIAN_TOKEN_RE = re.compile(r"^[а-я]+$")
QUERY_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]
ITEM_COLUMNS = [
    "item_id",
    "item_title_raw",
    "item_infm_params_text",
    "item_description_raw",
    "item_category_id",
    "item_location_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("assets/dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--validation-size", type=int, default=VALIDATION_SIZE)
    return parser.parse_args()


def normalize_text(text: object) -> str:
    """Apply only normalization that is shared by word and character branches."""
    if text is None or pd.isna(text):
        return ""
    return " ".join(str(text).lower().replace("ё", "е").split())


class TokenNormalizer:
    """Keep tokenization fixed while changing only the morphological transform."""

    def __init__(self, mode: str) -> None:
        if mode not in NORMALIZATIONS:
            raise ValueError(f"Unknown normalization: {mode}")
        self.mode = mode
        self.cache: dict[str, str] = {}
        self.stemmer = (
            snowballstemmer.stemmer("russian")
            if mode == "russian_snowball"
            else None
        )
        self.morph = pymorphy3.MorphAnalyzer() if mode == "russian_lemma" else None

    def _normalize_unseen(self, tokens: list[str]) -> None:
        if self.mode == "russian_snowball":
            assert self.stemmer is not None
            self.cache.update(zip(tokens, self.stemmer.stemWords(tokens)))
            return
        if self.mode == "russian_lemma":
            assert self.morph is not None
            for token in tokens:
                if RUSSIAN_TOKEN_RE.fullmatch(token):
                    lemma = self.morph.parse(token)[0].normal_form.replace("ё", "е")
                else:
                    lemma = token
                self.cache[token] = lemma
            return
        self.cache.update((token, token) for token in tokens)

    def __call__(self, text: object) -> str:
        tokens = TOKEN_RE.findall(normalize_text(text))
        unseen = list(dict.fromkeys(token for token in tokens if token not in self.cache))
        if unseen:
            self._normalize_unseen(unseen)
        return " ".join(self.cache[token] for token in tokens)


def stable_hash(value: str) -> str:
    return hashlib.sha1(f"{SEED}|{value}".encode()).hexdigest()


def signature_key(row: pd.Series) -> tuple[object, ...]:
    return tuple(row[column] for column in QUERY_COLUMNS)


def build_validation(
    train: pd.DataFrame, mode: str, size: int
) -> tuple[pd.DataFrame, dict[tuple[object, ...], set[str]]]:
    """Reproduce the fixed cold/warm proxy protocol used since experiment 0003."""
    grouped = train.groupby(QUERY_COLUMNS, sort=False, dropna=False)["item_id"].agg(
        lambda values: set(values)
    )
    signatures_by_text: dict[str, list[tuple[object, ...]]] = defaultdict(list)
    for key in grouped.index:
        signatures_by_text[normalize_text(key[0])].append(key)
    if mode == "cold":
        texts = list(signatures_by_text)
    elif mode == "warm":
        texts = [text for text, keys in signatures_by_text.items() if len(keys) >= 2]
    else:
        raise ValueError(mode)
    selected_texts = sorted(texts, key=stable_hash)[:size]
    selected_keys = [
        min(
            signatures_by_text[text],
            key=lambda key: stable_hash(repr(tuple(str(value) for value in key))),
        )
        for text in selected_texts
    ]
    validation = pd.DataFrame(selected_keys, columns=QUERY_COLUMNS)
    return validation, {key: grouped.loc[key] for key in selected_keys}


def target_category(value: object) -> int:
    category = int(value)
    return 114 if category == 0 else category


def preprocess_series(
    values: pd.Series, preprocessor: Callable[[object], str], label: str
) -> pd.Series:
    """Preprocess in visible chunks so a long lemmatization run reports progress."""
    started = time.monotonic()
    result: list[str] = []
    chunk_size = 25_000
    total = len(values)
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        result.extend(preprocessor(value) for value in values.iloc[start:stop])
        print(f"  {label}: {stop:,}/{total:,}", flush=True)
    print(f"  {label} finished in {time.monotonic() - started:.1f}s", flush=True)
    return pd.Series(result, index=values.index, dtype="object")


def build_word_documents(
    items: pd.DataFrame, preprocessor: TokenNormalizer
) -> pd.Series:
    title = preprocess_series(items["item_title_raw"], preprocessor, "item titles")
    params = preprocess_series(items["item_infm_params_text"], preprocessor, "item params")
    description = preprocess_series(
        items["item_description_raw"], preprocessor, "item descriptions"
    )
    documents = title + " " + title + " " + title + " " + params + " " + description
    del title, params, description
    return documents


def build_word_queries(
    validation: pd.DataFrame, preprocessor: TokenNormalizer, label: str
) -> pd.Series:
    query = preprocess_series(
        validation["search_query"], preprocessor, f"{label} query text"
    )
    filters = preprocess_series(
        validation["search_infm_params_text"], preprocessor, f"{label} query filters"
    )
    result = query + " " + query + " " + filters
    del query, filters
    return result


def top_sparse(
    scores, category: int, item_categories: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    indices = scores.indices
    values = scores.data
    if len(indices):
        keep = item_categories[indices] == category
        indices = indices[keep]
        values = values[keep]
    if len(indices) > CANDIDATE_POOL:
        selected = np.argpartition(values, -CANDIDATE_POOL)[-CANDIDATE_POOL:]
        indices = indices[selected]
        values = values[selected]
    return indices.astype(np.int32, copy=False), values.astype(np.float32, copy=False)


def retrieve(
    query_matrix,
    transposed_index,
    validation: pd.DataFrame,
    item_categories: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    for row_number, category_value in enumerate(validation["search_category"]):
        scores = (query_matrix[row_number] @ transposed_index).tocsr()
        rows.append(top_sparse(scores, target_category(category_value), item_categories))
    return rows


def build_fallbacks(
    items: pd.DataFrame,
) -> tuple[dict[int, np.ndarray], dict[tuple[int, int], np.ndarray]]:
    by_category: dict[int, list[int]] = defaultdict(list)
    by_category_location: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in items[["item_category_id", "item_location_id"]].iterrows():
        category = int(row["item_category_id"])
        location = int(row["item_location_id"])
        by_category[category].append(index)
        by_category_location[(category, location)].append(index)
    return (
        {key: np.asarray(value, dtype=np.int32) for key, value in by_category.items()},
        {
            key: np.asarray(value, dtype=np.int32)
            for key, value in by_category_location.items()
        },
    )


def fallback_candidates(
    category: int,
    location: int,
    category_fallbacks: dict[int, np.ndarray],
    location_fallbacks: dict[tuple[int, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    location_rows = location_fallbacks.get((category, location))
    if location_rows is not None and len(location_rows):
        indices = location_rows[:CANDIDATE_POOL]
    else:
        indices = category_fallbacks.get(category, np.empty(0, dtype=np.int32))[
            :CANDIDATE_POOL
        ]
    return indices, np.zeros(len(indices), dtype=np.float32)


def combine_candidates(
    word: tuple[np.ndarray, np.ndarray],
    char: tuple[np.ndarray, np.ndarray],
    char_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    word_indices, word_scores = word
    char_indices, char_scores = char
    if char_weight == 0.0:
        return word_indices, word_scores
    if char_weight == 1.0:
        return char_indices, char_scores
    indices = np.union1d(word_indices, char_indices).astype(np.int32, copy=False)
    scores = np.zeros(len(indices), dtype=np.float32)
    scores[np.searchsorted(indices, word_indices)] += (1.0 - char_weight) * word_scores
    scores[np.searchsorted(indices, char_indices)] += char_weight * char_scores
    return indices, scores


def evaluate_grid(
    normalization: str,
    word_ngram: str,
    mode: str,
    validation: pd.DataFrame,
    relevance_rows: dict[tuple[object, ...], frozenset[int]],
    word_retrieval: list[tuple[np.ndarray, np.ndarray]],
    char_retrievals: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    item_locations: np.ndarray,
    category_fallbacks: dict[int, np.ndarray],
    location_fallbacks: dict[tuple[int, int], np.ndarray],
) -> list[dict[str, object]]:
    sums = np.zeros(
        (len(CHAR_NGRAMS), len(CHAR_WEIGHTS), len(LOCATION_BOOSTS)), dtype=np.float64
    )
    queries_with_hit = np.zeros_like(sums, dtype=np.int32)
    char_names = list(CHAR_NGRAMS)

    for row_number, (_, query) in enumerate(validation.iterrows()):
        relevant = relevance_rows[signature_key(query)]
        denominator = len(relevant)
        category = target_category(query["search_category"])
        location = int(query["search_location_id"])
        word_candidates = word_retrieval[row_number]
        for char_number, char_name in enumerate(char_names):
            char_candidates = char_retrievals[char_name][row_number]
            for weight_number, weight_value in enumerate(CHAR_WEIGHTS):
                indices, text_scores = combine_candidates(
                    word_candidates, char_candidates, float(weight_value)
                )
                if not len(indices):
                    indices, text_scores = fallback_candidates(
                        category,
                        location,
                        category_fallbacks,
                        location_fallbacks,
                    )
                if not len(indices):
                    continue
                location_match = (
                    item_locations[indices] == location
                ).astype(np.float32)
                score_grid = text_scores[:, None] + (
                    location_match[:, None]
                    * np.asarray(LOCATION_BOOSTS, dtype=np.float32)[None, :]
                )
                take = min(TOP_K, len(indices))
                selected_local = np.argpartition(
                    score_grid, len(indices) - take, axis=0
                )[-take:, :]
                for boost_number in range(len(LOCATION_BOOSTS)):
                    selected = indices[selected_local[:, boost_number]]
                    hit_count = sum(int(item_row in relevant) for item_row in selected)
                    recall = hit_count / denominator
                    sums[char_number, weight_number, boost_number] += recall
                    queries_with_hit[char_number, weight_number, boost_number] += int(
                        hit_count > 0
                    )

    rows: list[dict[str, object]] = []
    for char_number, char_name in enumerate(char_names):
        for weight_number, weight_value in enumerate(CHAR_WEIGHTS):
            for boost_number, boost_value in enumerate(LOCATION_BOOSTS):
                rows.append(
                    {
                        "normalization": normalization,
                        "word_ngram": word_ngram,
                        "char_ngram": char_name,
                        "char_weight": float(weight_value),
                        "location_boost": float(boost_value),
                        "mode": mode,
                        "queries": len(validation),
                        "recall_at_50": float(
                            sums[char_number, weight_number, boost_number]
                            / len(validation)
                        ),
                        "queries_with_hit": int(
                            queries_with_hit[
                                char_number, weight_number, boost_number
                            ]
                        ),
                    }
                )
    return rows


def build_char_retrievals(
    items: pd.DataFrame,
    validations: dict[str, pd.DataFrame],
    item_categories: np.ndarray,
) -> tuple[
    dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]], list[dict[str, object]]
]:
    titles = items["item_title_raw"].fillna("").map(normalize_text)
    retrievals: dict[
        str, dict[str, list[tuple[np.ndarray, np.ndarray]]]
    ] = {}
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(
        args.data_dir / "train.parquet", columns=QUERY_COLUMNS + ITEM_COLUMNS
    )
    items = train[ITEM_COLUMNS].drop_duplicates("item_id").reset_index(drop=True)
    item_categories = items["item_category_id"].to_numpy()
    item_locations = items["item_location_id"].to_numpy()
    item_row_by_id = dict(zip(items["item_id"], items.index))
    category_fallbacks, location_fallbacks = build_fallbacks(items)

    validations: dict[str, pd.DataFrame] = {}
    relevance_rows: dict[
        str, dict[tuple[object, ...], frozenset[int]]
    ] = {}
    for mode in ("cold", "warm"):
        validation, relevance_ids = build_validation(train, mode, args.validation_size)
        validations[mode] = validation
        relevance_rows[mode] = {
            key: frozenset(item_row_by_id[item_id] for item_id in item_ids)
            for key, item_ids in relevance_ids.items()
        }

    char_retrievals, char_diagnostics = build_char_retrievals(
        items, validations, item_categories
    )

    result_rows: list[dict[str, object]] = []
    word_diagnostics: list[dict[str, object]] = []
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
                result_rows.extend(
                    evaluate_grid(
                        normalization,
                        word_name,
                        mode,
                        validation,
                        relevance_rows[mode],
                        word_retrieval,
                        mode_char_retrievals,
                        item_locations,
                        category_fallbacks,
                        location_fallbacks,
                    )
                )
                del query_matrix, word_retrieval
            word_diagnostics.append(details)
            del vectorizer, transposed
            gc.collect()
        del documents, query_documents, preprocessor
        gc.collect()

    metrics = pd.DataFrame(result_rows)
    config_columns = [
        "normalization",
        "word_ngram",
        "char_ngram",
        "char_weight",
        "location_boost",
    ]
    comparison = metrics.pivot_table(
        index=config_columns, columns="mode", values="recall_at_50"
    ).reset_index()
    comparison["benchmark_mix_63_cold_37_warm"] = (
        0.63 * comparison["cold"] + 0.37 * comparison["warm"]
    )
    comparison = comparison.sort_values(
        config_columns, kind="stable"
    ).reset_index(drop=True)
    metrics = metrics.merge(
        comparison[config_columns + ["benchmark_mix_63_cold_37_warm"]],
        on=config_columns,
        how="left",
    )
    metrics = metrics.sort_values(config_columns + ["mode"], kind="stable")
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)

    ranked = comparison.sort_values(
        "benchmark_mix_63_cold_37_warm", ascending=False, kind="stable"
    ).reset_index(drop=True)
    report = {
        "grid": {
            "normalizations": list(NORMALIZATIONS),
            "word_ngrams": {key: list(value) for key, value in WORD_NGRAMS.items()},
            "char_ngrams": {key: list(value) for key, value in CHAR_NGRAMS.items()},
            "char_weights": CHAR_WEIGHTS,
            "location_boosts": LOCATION_BOOSTS,
            "configurations": len(comparison),
        },
        "validation": {
            "cold_queries": len(validations["cold"]),
            "warm_queries": len(validations["warm"]),
            "candidate_pool": CANDIDATE_POOL,
            "top_k": TOP_K,
            "mixture": {"cold": 0.63, "warm": 0.37},
        },
        "libraries": {
            "scikit-learn": importlib.metadata.version("scikit-learn"),
            "snowballstemmer": importlib.metadata.version("snowballstemmer"),
            "pymorphy3": importlib.metadata.version("pymorphy3"),
            "pymorphy3-dicts-ru": importlib.metadata.version("pymorphy3-dicts-ru"),
        },
        "char_indexes": char_diagnostics,
        "word_indexes": word_diagnostics,
        "best_config": ranked.iloc[0].to_dict(),
        "top_configs": ranked.head(20).to_dict("records"),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nTop configurations:", flush=True)
    print(ranked.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
