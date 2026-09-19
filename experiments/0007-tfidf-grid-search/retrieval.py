"""Sparse candidate retrieval, fallbacks, and word/char score blending."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from settings import CANDIDATE_POOL
from validation import target_category


CandidateRow = tuple[np.ndarray, np.ndarray]
CandidateRows = list[CandidateRow]


def top_sparse(
    scores, category: int, item_categories: np.ndarray
) -> CandidateRow:
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
) -> CandidateRows:
    rows: CandidateRows = []
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
) -> CandidateRow:
    location_rows = location_fallbacks.get((category, location))
    if location_rows is not None and len(location_rows):
        indices = location_rows[:CANDIDATE_POOL]
    else:
        indices = category_fallbacks.get(category, np.empty(0, dtype=np.int32))[
            :CANDIDATE_POOL
        ]
    return indices, np.zeros(len(indices), dtype=np.float32)


def combine_candidates(
    word: CandidateRow, char: CandidateRow, char_weight: float
) -> CandidateRow:
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
