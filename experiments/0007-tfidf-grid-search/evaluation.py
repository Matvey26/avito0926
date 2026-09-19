"""Recall@50 evaluation for the complete score grid and bootstrap samples."""

from __future__ import annotations

import numpy as np
import pandas as pd

from retrieval import CandidateRows, combine_candidates, fallback_candidates
from settings import CHAR_NGRAMS, CHAR_WEIGHTS, LOCATION_BOOSTS, TOP_K
from validation import signature_key, target_category


def evaluate_grid(
    normalization: str,
    word_ngram: str,
    mode: str,
    validation: pd.DataFrame,
    relevance_rows: dict[tuple[object, ...], frozenset[int]],
    bootstrap_weights: np.ndarray,
    word_retrieval: CandidateRows,
    char_retrievals: dict[str, CandidateRows],
    item_locations: np.ndarray,
    category_fallbacks: dict[int, np.ndarray],
    location_fallbacks: dict[tuple[int, int], np.ndarray],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grid_shape = (len(CHAR_NGRAMS), len(CHAR_WEIGHTS), len(LOCATION_BOOSTS))
    sums = np.zeros(grid_shape, dtype=np.float64)
    bootstrap_sums = np.zeros((len(bootstrap_weights), *grid_shape), dtype=np.float64)
    queries_with_hit = np.zeros(grid_shape, dtype=np.int32)
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
                recalls = np.empty(len(LOCATION_BOOSTS), dtype=np.float64)
                for boost_number in range(len(LOCATION_BOOSTS)):
                    selected = indices[selected_local[:, boost_number]]
                    hit_count = sum(int(item_row in relevant) for item_row in selected)
                    recalls[boost_number] = hit_count / denominator
                    queries_with_hit[char_number, weight_number, boost_number] += int(
                        hit_count > 0
                    )
                sums[char_number, weight_number] += recalls
                bootstrap_sums[:, char_number, weight_number] += (
                    bootstrap_weights[:, row_number, None] * recalls[None, :]
                )

    rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    for char_number, char_name in enumerate(char_names):
        for weight_number, weight_value in enumerate(CHAR_WEIGHTS):
            for boost_number, boost_value in enumerate(LOCATION_BOOSTS):
                config = {
                    "normalization": normalization,
                    "word_ngram": word_ngram,
                    "char_ngram": char_name,
                    "char_weight": float(weight_value),
                    "location_boost": float(boost_value),
                    "mode": mode,
                }
                rows.append(
                    {
                        **config,
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
                for bootstrap_run in range(len(bootstrap_weights)):
                    bootstrap_rows.append(
                        {
                            **config,
                            "bootstrap_run": bootstrap_run,
                            "recall_at_50": float(
                                bootstrap_sums[
                                    bootstrap_run,
                                    char_number,
                                    weight_number,
                                    boost_number,
                                ]
                                / len(validation)
                            ),
                        }
                    )
    return rows, bootstrap_rows
