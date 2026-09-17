#!/usr/bin/env python3
"""Evaluate popularity and exact-query-history candidate generators."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SEED = "20260918"
VALIDATION_SIZE = 2_500
TOP_K = 50
QUERY_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("assets/dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--validation-size", type=int, default=VALIDATION_SIZE)
    return parser.parse_args()


def normalize(text: object) -> str:
    return " ".join(str(text or "").lower().replace("ё", "е").split())


def stable_hash(value: str) -> str:
    return hashlib.sha1(f"{SEED}|{value}".encode()).hexdigest()


def signature_key(row: pd.Series) -> tuple[object, ...]:
    return tuple(row[column] for column in QUERY_COLUMNS)


def signature_text(key: tuple[object, ...]) -> str:
    return normalize(key[0])


def build_validation(
    train: pd.DataFrame, mode: str, size: int
) -> tuple[pd.DataFrame, dict[tuple[object, ...], set[str]]]:
    grouped = train.groupby(QUERY_COLUMNS, sort=False, dropna=False)["item_id"].agg(
        lambda values: set(values)
    )
    signatures_by_text: dict[str, list[tuple[object, ...]]] = defaultdict(list)
    for key in grouped.index:
        signatures_by_text[signature_text(key)].append(key)

    if mode == "cold":
        candidate_texts = list(signatures_by_text)
    elif mode == "warm":
        candidate_texts = [
            text for text, signatures in signatures_by_text.items() if len(signatures) >= 2
        ]
    else:
        raise ValueError(f"Unknown validation mode: {mode}")

    selected_texts = sorted(candidate_texts, key=stable_hash)[:size]
    selected_keys = []
    for text in selected_texts:
        keys = signatures_by_text[text]
        selected_keys.append(
            min(keys, key=lambda key: stable_hash(repr(tuple(str(v) for v in key))))
        )

    validation = pd.DataFrame(selected_keys, columns=QUERY_COLUMNS)
    relevance = {key: grouped.loc[key] for key in selected_keys}
    return validation, relevance


def remove_validation_history(
    train: pd.DataFrame, validation: pd.DataFrame, mode: str
) -> pd.DataFrame:
    if mode == "cold":
        held_out_texts = set(validation["search_query"].map(normalize))
        return train.loc[~train["query_norm"].isin(held_out_texts)].copy()

    held_out = pd.MultiIndex.from_frame(validation[QUERY_COLUMNS])
    train_index = pd.MultiIndex.from_frame(train[QUERY_COLUMNS])
    return train.loc[~train_index.isin(held_out)].copy()


def ranked_ids(frame: pd.DataFrame, group_columns: list[str]) -> dict[object, list[str]]:
    counts = (
        frame.groupby(group_columns + ["item_id"], sort=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(group_columns + ["count", "item_id"], ascending=[True] * len(group_columns) + [False, True])
    )
    if not group_columns:
        return {"all": counts["item_id"].head(TOP_K).tolist()}
    result: dict[object, list[str]] = {}
    for key, group in counts.groupby(group_columns, sort=False):
        result[key] = group["item_id"].head(TOP_K).tolist()
    return result


def extend_unique(target: list[str], candidates: Iterable[str]) -> None:
    seen = set(target)
    for item_id in candidates:
        if item_id not in seen:
            target.append(item_id)
            seen.add(item_id)
            if len(target) == TOP_K:
                return


def predictions_for_query(
    query: pd.Series,
    method: str,
    global_rank: list[str],
    category_rank: dict[object, list[str]],
    location_rank: dict[object, list[str]],
    query_rank: dict[object, list[str]],
) -> list[str]:
    target_category = 114 if int(query["search_category"]) == 0 else int(query["search_category"])
    category_candidates = category_rank.get(target_category, [])
    location_key = (target_category, int(query["search_location_id"]))
    query_candidates = query_rank.get(normalize(query["search_query"]), [])
    prediction: list[str] = []

    if method == "global_popularity":
        extend_unique(prediction, global_rank)
    elif method == "category_popularity":
        extend_unique(prediction, category_candidates)
        extend_unique(prediction, global_rank)
    elif method == "location_popularity":
        extend_unique(prediction, location_rank.get(location_key, []))
        extend_unique(prediction, category_candidates)
        extend_unique(prediction, global_rank)
    elif method == "query_history":
        extend_unique(prediction, query_candidates)
        extend_unique(prediction, location_rank.get(location_key, []))
        extend_unique(prediction, category_candidates)
        extend_unique(prediction, global_rank)
    else:
        raise ValueError(method)
    return prediction[:TOP_K]


def evaluate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    relevance: dict[tuple[object, ...], set[str]],
    mode: str,
) -> list[dict[str, object]]:
    history = remove_validation_history(train, validation, mode)
    global_rank = ranked_ids(history, [])["all"]
    category_rank = ranked_ids(history, ["item_category_id"])
    location_rank = ranked_ids(history, ["item_category_id", "item_location_id"])
    query_rank = ranked_ids(history, ["query_norm"])
    methods = [
        "global_popularity",
        "category_popularity",
        "location_popularity",
        "query_history",
    ]
    recalls = {method: [] for method in methods}

    for _, query in validation.iterrows():
        relevant = relevance[signature_key(query)]
        for method in methods:
            predicted = predictions_for_query(
                query,
                method,
                global_rank,
                category_rank,
                location_rank,
                query_rank,
            )
            recalls[method].append(len(set(predicted) & relevant) / len(relevant))

    return [
        {
            "mode": mode,
            "method": method,
            "queries": len(validation),
            "recall_at_50": float(np.mean(values)),
            "queries_with_hit": int(np.count_nonzero(values)),
        }
        for method, values in recalls.items()
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(
        args.data_dir / "train.parquet",
        columns=QUERY_COLUMNS
        + ["item_id", "item_category_id", "item_location_id"],
    )
    train["query_norm"] = train["search_query"].map(normalize)

    results: list[dict[str, object]] = []
    split_report: dict[str, object] = {}
    for mode in ("cold", "warm"):
        validation, relevance = build_validation(train, mode, args.validation_size)
        results.extend(evaluate(train, validation, relevance, mode))
        split_report[mode] = {
            "queries": len(validation),
            "unique_normalized_texts": int(
                validation["search_query"].map(normalize).nunique()
            ),
            "median_relevant_items": float(
                np.median([len(items) for items in relevance.values()])
            ),
        }

    metrics = pd.DataFrame(results)
    pivot = metrics.pivot(index="method", columns="mode", values="recall_at_50")
    pivot["benchmark_mix_63_cold_37_warm"] = 0.63 * pivot["cold"] + 0.37 * pivot["warm"]
    metrics = metrics.merge(
        pivot["benchmark_mix_63_cold_37_warm"], on="method", how="left"
    )
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    (args.output_dir / "report.json").write_text(
        json.dumps({"splits": split_report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
