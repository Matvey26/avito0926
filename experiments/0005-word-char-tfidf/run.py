#!/usr/bin/env python3
"""Evaluate a simple union of word/full-text and char/title TF-IDF candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


SEED = "20260918"
VALIDATION_SIZE = 2_500
TOP_K = 50
CANDIDATE_POOL = 500
CHAR_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
LOCATION_BOOSTS = [0.3, 0.5, 0.7]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("assets/dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--validation-size", type=int, default=VALIDATION_SIZE)
    return parser.parse_args()


def normalize(text: object) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return " ".join(str(text).lower().replace("ё", "е").split())


def stable_hash(value: str) -> str:
    return hashlib.sha1(f"{SEED}|{value}".encode()).hexdigest()


def signature_key(row: pd.Series) -> tuple[object, ...]:
    return tuple(row[column] for column in QUERY_COLUMNS)


def build_validation(
    train: pd.DataFrame, mode: str, size: int
) -> tuple[pd.DataFrame, dict[tuple[object, ...], set[str]]]:
    grouped = train.groupby(QUERY_COLUMNS, sort=False, dropna=False)["item_id"].agg(
        lambda values: set(values)
    )
    signatures_by_text: dict[str, list[tuple[object, ...]]] = defaultdict(list)
    for key in grouped.index:
        signatures_by_text[normalize(key[0])].append(key)
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


def full_item_text(items: pd.DataFrame) -> pd.Series:
    title = items["item_title_raw"].fillna("").map(normalize)
    params = items["item_infm_params_text"].fillna("").map(normalize)
    description = items["item_description_raw"].fillna("").map(normalize)
    return title + " " + title + " " + title + " " + params + " " + description


def full_query_text(queries: pd.DataFrame) -> pd.Series:
    text = queries["search_query"].fillna("").map(normalize)
    filters = queries["search_infm_params_text"].fillna("").map(normalize)
    return text + " " + text + " " + filters


def target_category(value: object) -> int:
    category = int(value)
    return 114 if category == 0 else category


def top_sparse(scores, eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = scores.indices
    values = scores.data
    if len(indices):
        keep = eligible[indices]
        indices = indices[keep]
        values = values[keep]
    if len(indices) > CANDIDATE_POOL:
        selected = np.argpartition(values, -CANDIDATE_POOL)[-CANDIDATE_POOL:]
        indices = indices[selected]
        values = values[selected]
    return indices, values


def combine_scores(
    word_indices: np.ndarray,
    word_scores: np.ndarray,
    char_indices: np.ndarray,
    char_scores: np.ndarray,
    char_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    if char_weight == 0:
        return word_indices, word_scores.copy()
    indices = np.union1d(word_indices, char_indices)
    scores = np.zeros(len(indices), dtype=np.float32)
    scores[np.searchsorted(indices, word_indices)] += word_scores
    scores[np.searchsorted(indices, char_indices)] += char_weight * char_scores
    return indices, scores


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(
        args.data_dir / "train.parquet", columns=QUERY_COLUMNS + ITEM_COLUMNS
    )
    items = train[ITEM_COLUMNS].drop_duplicates("item_id").reset_index(drop=True)
    validations: dict[str, pd.DataFrame] = {}
    relevances: dict[str, dict[tuple[object, ...], set[str]]] = {}
    for mode in ("cold", "warm"):
        validations[mode], relevances[mode] = build_validation(
            train, mode, args.validation_size
        )

    print("Fitting word/full-text index...", flush=True)
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.99,
        max_features=200_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    word_matrix = word_vectorizer.fit_transform(full_item_text(items))
    word_shape = list(word_matrix.shape)
    word_nnz = int(word_matrix.nnz)
    word_transposed = word_matrix.T.tocsr()
    del word_matrix

    print("Fitting char/title index...", flush=True)
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=150_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    titles = items["item_title_raw"].fillna("").map(normalize)
    char_matrix = char_vectorizer.fit_transform(titles)
    char_shape = list(char_matrix.shape)
    char_nnz = int(char_matrix.nnz)
    char_transposed = char_matrix.T.tocsr()
    del char_matrix

    item_ids = items["item_id"].to_numpy()
    item_locations = items["item_location_id"].to_numpy()
    item_categories = items["item_category_id"].to_numpy()
    result_rows: list[dict[str, object]] = []

    for mode, validation in validations.items():
        print(f"Evaluating {mode}...", flush=True)
        word_queries = word_vectorizer.transform(full_query_text(validation))
        char_queries = char_vectorizer.transform(
            validation["search_query"].fillna("").map(normalize)
        )
        recalls = {
            (char_weight, location_boost): []
            for char_weight in CHAR_WEIGHTS
            for location_boost in LOCATION_BOOSTS
        }
        for row_number, (_, query) in enumerate(validation.iterrows()):
            eligible = item_categories == target_category(query["search_category"])
            word_indices, word_scores = top_sparse(
                (word_queries[row_number] @ word_transposed).tocsr(), eligible
            )
            char_indices, char_scores = top_sparse(
                (char_queries[row_number] @ char_transposed).tocsr(), eligible
            )
            relevant = relevances[mode][signature_key(query)]
            for char_weight in CHAR_WEIGHTS:
                indices, text_scores = combine_scores(
                    word_indices,
                    word_scores,
                    char_indices,
                    char_scores,
                    char_weight,
                )
                location_match = (
                    item_locations[indices] == int(query["search_location_id"])
                ).astype(np.float32)
                for location_boost in LOCATION_BOOSTS:
                    scores = text_scores + location_boost * location_match
                    order = np.lexsort((item_ids[indices], -scores))[:TOP_K]
                    predicted = set(item_ids[indices[order]])
                    recalls[(char_weight, location_boost)].append(
                        len(predicted & relevant) / len(relevant)
                    )

        for (char_weight, location_boost), values in recalls.items():
            result_rows.append(
                {
                    "mode": mode,
                    "char_weight": char_weight,
                    "location_boost": location_boost,
                    "queries": len(validation),
                    "recall_at_50": float(np.mean(values)),
                    "queries_with_hit": int(np.count_nonzero(values)),
                }
            )

    metrics = pd.DataFrame(result_rows)
    pivot = metrics.pivot_table(
        index=["char_weight", "location_boost"],
        columns="mode",
        values="recall_at_50",
    )
    pivot["benchmark_mix_63_cold_37_warm"] = 0.63 * pivot["cold"] + 0.37 * pivot["warm"]
    metrics = metrics.merge(
        pivot["benchmark_mix_63_cold_37_warm"],
        on=["char_weight", "location_boost"],
        how="left",
    )
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    report = {
        "word_index": {"shape": word_shape, "nnz": word_nnz},
        "char_index": {"shape": char_shape, "nnz": char_nnz},
        "top_configs": pivot.reset_index()
        .sort_values("benchmark_mix_63_cold_37_warm", ascending=False)
        .head(10)
        .to_dict("records"),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pivot.sort_values("benchmark_mix_63_cold_37_warm", ascending=False).head(10))


if __name__ == "__main__":
    main()
