#!/usr/bin/env python3
"""Generate answer.csv with a deterministic word/char TF-IDF retriever."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


TOP_K = 50
CANDIDATE_POOL = 500
CHAR_WEIGHT = 0.25
LOCATION_BOOST = 0.7
WORD_MAX_FEATURES = 200_000
CHAR_MAX_FEATURES = 150_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("assets/dataset"))
    parser.add_argument("--output", type=Path, default=Path("answer.csv"))
    return parser.parse_args()


def normalize(text: object) -> str:
    """Apply a small deterministic normalization shared by both indexes."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return " ".join(str(text).lower().replace("ё", "е").split())


def build_word_item_text(items: pd.DataFrame) -> pd.Series:
    """Give the precise title more weight while retaining all available text."""
    title = items["item_title_raw"].fillna("").map(normalize)
    params = items["item_infm_params_text"].fillna("").map(normalize)
    description = items["item_description_raw"].fillna("").map(normalize)
    return title + " " + title + " " + title + " " + params + " " + description


def build_word_query_text(queries: pd.DataFrame) -> pd.Series:
    """Query filters are useful but the typed query remains the primary signal."""
    query = queries["search_query"].fillna("").map(normalize)
    filters = queries["search_infm_params_text"].fillna("").map(normalize)
    return query + " " + query + " " + filters


def target_category(search_category: object) -> int:
    """EDA showed that category zero is an unspecified services category."""
    category = int(search_category)
    return 114 if category == 0 else category


def deterministic_top_sparse(
    sparse_scores,
    eligible: np.ndarray,
    item_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a bounded sparse candidate set with stable item-id tie-breaking."""
    indices = sparse_scores.indices
    values = sparse_scores.data
    if len(indices):
        keep = eligible[indices]
        indices = indices[keep]
        values = values[keep]
    if len(indices) <= CANDIDATE_POOL:
        return indices, values

    threshold = np.partition(values, -CANDIDATE_POOL)[-CANDIDATE_POOL]
    above = np.flatnonzero(values > threshold)
    remaining = CANDIDATE_POOL - len(above)
    tied = np.flatnonzero(values == threshold)
    tied_order = np.argsort(item_ids[indices[tied]], kind="stable")[:remaining]
    selected = np.concatenate([above, tied[tied_order]])
    return indices[selected], values[selected]


def combine_scores(
    word_indices: np.ndarray,
    word_scores: np.ndarray,
    char_indices: np.ndarray,
    char_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge both sparse candidate pools into one weighted score vector."""
    indices = np.union1d(word_indices, char_indices)
    scores = np.zeros(len(indices), dtype=np.float32)
    scores[np.searchsorted(indices, word_indices)] += word_scores
    scores[np.searchsorted(indices, char_indices)] += CHAR_WEIGHT * char_scores
    return indices, scores


def build_fallbacks(items: pd.DataFrame) -> tuple[dict[tuple[int, int], list[int]], dict[int, list[int]]]:
    """Create deterministic quality-oriented fillers for rare zero-match queries."""
    ranked = items.assign(
        reviews=items["item_rating_reviews_count"].fillna(-1),
        rating=items["item_rating"].fillna(-1),
    ).sort_values(
        ["reviews", "rating", "item_id"], ascending=[False, False, True]
    )
    by_location = {
        (int(category), int(location)): group.index.tolist()
        for (category, location), group in ranked.groupby(
            ["item_category_id", "item_location_id"], sort=False
        )
    }
    by_category = {
        int(category): group.index.tolist()
        for category, group in ranked.groupby("item_category_id", sort=False)
    }
    return by_location, by_category


def fill_candidates(
    selected: list[int],
    fallbacks: list[list[int]],
) -> list[int]:
    """Fill to TOP_K without duplicates while preserving the ranked prefix."""
    seen = set(selected)
    for fallback in fallbacks:
        for item_index in fallback:
            if item_index not in seen:
                selected.append(item_index)
                seen.add(item_index)
                if len(selected) == TOP_K:
                    return selected
    return selected


def validate_submission(
    answer: pd.DataFrame,
    queries: pd.DataFrame,
    valid_item_ids: set[str],
) -> None:
    """Fail loudly on every submission-format invariant from TASK.md."""
    if answer.columns.tolist() != ["query_id", "answer"]:
        raise ValueError("Submission must contain exactly query_id and answer")
    if len(answer) != len(queries) or answer["query_id"].duplicated().any():
        raise ValueError("Submission must contain one unique row per query")
    if set(answer["query_id"]) != set(queries["query_id"]):
        raise ValueError("Submission query_id set differs from benchmark queries")
    if not answer["query_id"].str.len().eq(16).all():
        raise ValueError("Every query_id must have length 16")

    for query_id, serialized in answer.itertuples(index=False):
        item_ids = serialized.split()
        if not 1 <= len(item_ids) <= TOP_K:
            raise ValueError(f"{query_id}: expected 1..{TOP_K} item ids")
        if len(item_ids) != len(set(item_ids)):
            raise ValueError(f"{query_id}: duplicate item ids")
        if not set(item_ids) <= valid_item_ids:
            raise ValueError(f"{query_id}: unknown item id")


def main() -> None:
    args = parse_args()
    items = pd.read_parquet(args.data_dir / "benchmark_items.parquet").reset_index(
        drop=True
    )
    queries = pd.read_parquet(args.data_dir / "benchmark_queries.parquet")
    item_ids = items["item_id"].to_numpy()
    item_locations = items["item_location_id"].to_numpy()
    item_categories = items["item_category_id"].to_numpy()

    print("Building word/full-text TF-IDF index...", flush=True)
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.99,
        max_features=WORD_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )
    word_matrix = word_vectorizer.fit_transform(build_word_item_text(items))
    word_transposed = word_matrix.T.tocsr()
    del word_matrix

    print("Building char/title TF-IDF index...", flush=True)
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=CHAR_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char_matrix = char_vectorizer.fit_transform(
        items["item_title_raw"].fillna("").map(normalize)
    )
    char_transposed = char_matrix.T.tocsr()
    del char_matrix

    word_queries = word_vectorizer.transform(build_word_query_text(queries))
    char_queries = char_vectorizer.transform(
        queries["search_query"].fillna("").map(normalize)
    )
    location_fallback, category_fallback = build_fallbacks(items)

    predictions: list[list[str]] = []
    print(f"Retrieving candidates for {len(queries)} queries...", flush=True)
    for row_number, query in queries.iterrows():
        category = target_category(query["search_category"])
        location = int(query["search_location_id"])
        eligible = item_categories == category
        word_indices, word_scores = deterministic_top_sparse(
            (word_queries[row_number] @ word_transposed).tocsr(),
            eligible,
            item_ids,
        )
        char_indices, char_scores = deterministic_top_sparse(
            (char_queries[row_number] @ char_transposed).tocsr(),
            eligible,
            item_ids,
        )
        indices, scores = combine_scores(
            word_indices, word_scores, char_indices, char_scores
        )
        scores += LOCATION_BOOST * (item_locations[indices] == location)
        order = np.lexsort((item_ids[indices], -scores))[:TOP_K]
        selected = indices[order].tolist()
        selected = fill_candidates(
            selected,
            [
                location_fallback.get((category, location), []),
                category_fallback.get(category, []),
            ],
        )
        predictions.append(item_ids[selected[:TOP_K]].tolist())

    answer = pd.DataFrame(
        {
            "query_id": queries["query_id"],
            "answer": [" ".join(item_ids_for_query) for item_ids_for_query in predictions],
        }
    )
    validate_submission(answer, queries, set(item_ids))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    answer.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote validated submission to {args.output}")


if __name__ == "__main__":
    main()
