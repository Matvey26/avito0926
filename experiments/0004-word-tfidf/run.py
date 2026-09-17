#!/usr/bin/env python3
"""Evaluate compact word TF-IDF retrieval variants and location boosts."""

from __future__ import annotations

import argparse
import gc
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
LOCATION_BOOSTS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
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
        candidate_texts = list(signatures_by_text)
    elif mode == "warm":
        candidate_texts = [
            text for text, signatures in signatures_by_text.items() if len(signatures) >= 2
        ]
    else:
        raise ValueError(mode)
    selected_texts = sorted(candidate_texts, key=stable_hash)[:size]
    selected_keys = [
        min(
            signatures_by_text[text],
            key=lambda key: stable_hash(repr(tuple(str(value) for value in key))),
        )
        for text in selected_texts
    ]
    validation = pd.DataFrame(selected_keys, columns=QUERY_COLUMNS)
    return validation, {key: grouped.loc[key] for key in selected_keys}


def item_text(items: pd.DataFrame, variant: str) -> pd.Series:
    title = items["item_title_raw"].fillna("").map(normalize)
    if variant == "title":
        return title
    params = items["item_infm_params_text"].fillna("").map(normalize)
    if variant == "title_params":
        return title + " " + title + " " + params
    if variant == "all_text":
        description = items["item_description_raw"].fillna("").map(normalize)
        return title + " " + title + " " + title + " " + params + " " + description
    raise ValueError(variant)


def query_text(queries: pd.DataFrame, variant: str) -> pd.Series:
    text = queries["search_query"].fillna("").map(normalize)
    if variant == "title":
        return text
    filters = queries["search_infm_params_text"].fillna("").map(normalize)
    return text + " " + text + " " + filters


def target_category(value: object) -> int:
    category = int(value)
    return 114 if category == 0 else category


def top_pool(scores, eligible_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = scores.indices
    values = scores.data
    if len(indices):
        keep = eligible_mask[indices]
        indices = indices[keep]
        values = values[keep]
    if len(indices) > CANDIDATE_POOL:
        selected = np.argpartition(values, -CANDIDATE_POOL)[-CANDIDATE_POOL:]
        indices = indices[selected]
        values = values[selected]
    return indices, values


def evaluate_variant(
    variant: str,
    items: pd.DataFrame,
    validations: dict[str, pd.DataFrame],
    relevances: dict[str, dict[tuple[object, ...], set[str]]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.99,
        max_features=200_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    documents = item_text(items, variant)
    item_matrix = vectorizer.fit_transform(documents)
    matrix_shape = list(item_matrix.shape)
    matrix_nnz = int(item_matrix.nnz)
    # CSR @ CSC would make scipy convert the full transposed corpus on every
    # query. Materialize the CSR transpose once instead.
    item_matrix_transposed = item_matrix.T.tocsr()
    del item_matrix
    item_ids = items["item_id"].to_numpy()
    item_locations = items["item_location_id"].to_numpy()
    item_categories = items["item_category_id"].to_numpy()

    results: list[dict[str, object]] = []
    for mode, validation in validations.items():
        query_matrix = vectorizer.transform(query_text(validation, variant))
        recalls = {boost: [] for boost in LOCATION_BOOSTS}
        for row_number, (_, query) in enumerate(validation.iterrows()):
            category = target_category(query["search_category"])
            eligible = item_categories == category
            sparse_scores = (
                query_matrix[row_number] @ item_matrix_transposed
            ).tocsr()
            indices, lexical_scores = top_pool(sparse_scores, eligible)
            if not len(indices):
                indices = np.flatnonzero(
                    eligible & (item_locations == int(query["search_location_id"]))
                )[:TOP_K]
                lexical_scores = np.zeros(len(indices), dtype=np.float32)
            relevant = relevances[mode][signature_key(query)]
            location_matches = (
                item_locations[indices] == int(query["search_location_id"])
            ).astype(np.float32)
            for boost in LOCATION_BOOSTS:
                scores = lexical_scores + boost * location_matches
                take = min(TOP_K, len(indices))
                if take:
                    chosen_local = np.argpartition(scores, -take)[-take:]
                    predicted = set(item_ids[indices[chosen_local]])
                else:
                    predicted = set()
                recalls[boost].append(len(predicted & relevant) / len(relevant))

        for boost, values in recalls.items():
            results.append(
                {
                    "variant": variant,
                    "location_boost": boost,
                    "mode": mode,
                    "queries": len(validation),
                    "recall_at_50": float(np.mean(values)),
                    "queries_with_hit": int(np.count_nonzero(values)),
                }
            )

    diagnostics = {
        "variant": variant,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "item_matrix_shape": matrix_shape,
        "item_matrix_nnz": matrix_nnz,
    }
    del documents, item_matrix_transposed, vectorizer
    gc.collect()
    return results, diagnostics


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

    all_results: list[dict[str, object]] = []
    diagnostics = []
    for variant in ("title", "title_params", "all_text"):
        print(f"Evaluating {variant}...", flush=True)
        results, variant_diagnostics = evaluate_variant(
            variant, items, validations, relevances
        )
        all_results.extend(results)
        diagnostics.append(variant_diagnostics)

    metrics = pd.DataFrame(all_results)
    pivot = metrics.pivot_table(
        index=["variant", "location_boost"], columns="mode", values="recall_at_50"
    )
    pivot["benchmark_mix_63_cold_37_warm"] = 0.63 * pivot["cold"] + 0.37 * pivot["warm"]
    metrics = metrics.merge(
        pivot["benchmark_mix_63_cold_37_warm"],
        on=["variant", "location_boost"],
        how="left",
    )
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    best = (
        pivot.reset_index()
        .sort_values("benchmark_mix_63_cold_37_warm", ascending=False)
        .head(10)
        .to_dict("records")
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(
            {"diagnostics": diagnostics, "top_configs": best},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pivot.sort_values("benchmark_mix_63_cold_37_warm", ascending=False).head(12))


if __name__ == "__main__":
    main()
