#!/usr/bin/env python3
"""Measure simple relevance signals against category-matched random negatives."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260918
SAMPLE_SIZE = 50_000
TOKEN_RE = re.compile(r"(?u)\b[\w-]{2,}\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("assets/dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    return parser.parse_args()


def normalize(text: object) -> str:
    return " ".join(str(text or "").lower().replace("ё", "е").split())


def tokens(text: object) -> set[str]:
    return set(TOKEN_RE.findall(normalize(text)))


def coverage(needles: set[str], haystack: set[str]) -> float:
    return len(needles & haystack) / len(needles) if needles else 0.0


def pair_features(
    query: pd.Series,
    item: pd.Series,
    item_frequency: dict[str, int],
) -> dict[str, float]:
    query_text = normalize(query["search_query"])
    query_tokens = tokens(query_text)
    filter_tokens = tokens(query["search_infm_params_text"])
    title = normalize(item["item_title_raw"])
    params = normalize(item["item_infm_params_text"])
    description = normalize(item["item_description_raw"])
    title_tokens = tokens(title)
    param_tokens = tokens(params)
    description_tokens = tokens(description)
    all_tokens = title_tokens | param_tokens | description_tokens
    reviews = item["item_rating_reviews_count"]
    rating = item["item_rating"]

    return {
        "title_query_coverage": coverage(query_tokens, title_tokens),
        "params_query_coverage": coverage(query_tokens, param_tokens),
        "description_query_coverage": coverage(query_tokens, description_tokens),
        "all_text_query_coverage": coverage(query_tokens, all_tokens),
        "params_filter_coverage": coverage(filter_tokens, param_tokens),
        "exact_query_in_title": float(bool(query_text) and query_text in title),
        "exact_query_in_description": float(
            bool(query_text) and query_text in description
        ),
        "category_match": float(query["search_category"] == item["item_category_id"]),
        "location_match": float(query["search_location_id"] == item["item_location_id"]),
        "log_reviews": float(np.log1p(0.0 if pd.isna(reviews) else reviews)),
        "rating": float(0.0 if pd.isna(rating) else rating),
        "phone_visible": float(not item["item_is_phone_hidden"]),
        "messages_allowed": float(not item["item_is_message_forbidden"]),
        "log_train_click_frequency": float(
            np.log1p(item_frequency.get(item["item_id"], 0))
        ),
    }


def quantiles(values: pd.Series) -> dict[str, float]:
    return {
        str(q): float(value)
        for q, value in values.quantile([0, 0.5, 0.9, 0.99, 1]).items()
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.data_dir / "train.parquet"
    item_path = args.data_dir / "benchmark_items.parquet"
    query_path = args.data_dir / "benchmark_queries.parquet"

    columns = [
        "search_query",
        "search_location_id",
        "search_infm_params_text",
        "search_category",
        "item_id",
        "item_title_raw",
        "item_infm_params_text",
        "item_description_raw",
        "item_category_id",
        "item_location_id",
        "item_rating",
        "item_rating_reviews_count",
        "item_is_phone_hidden",
        "item_is_message_forbidden",
    ]
    train = pd.read_parquet(train_path, columns=columns)
    benchmark_items = pd.read_parquet(
        item_path, columns=["item_id", "item_category_id"]
    )
    benchmark_queries = pd.read_parquet(query_path)

    sample_size = min(args.sample_size, len(train))
    positives = train.sample(sample_size, random_state=SEED).reset_index(drop=True)
    item_pool = train.drop_duplicates("item_id").reset_index(drop=True)
    pools = {
        int(category): group.reset_index(drop=True)
        for category, group in item_pool.groupby("item_category_id")
    }
    default_pool = pools[114]
    rng = np.random.default_rng(SEED)

    negative_rows: list[pd.Series] = []
    for _, positive in positives.iterrows():
        pool = pools.get(int(positive["search_category"]), default_pool)
        candidate = pool.iloc[int(rng.integers(0, len(pool)))]
        if candidate["item_id"] == positive["item_id"]:
            candidate = pool.iloc[(candidate.name + 1) % len(pool)]
        negative_rows.append(candidate)
    negatives = pd.DataFrame(negative_rows).reset_index(drop=True)

    item_frequency = train["item_id"].value_counts().to_dict()
    records: list[dict[str, float]] = []
    for index in range(sample_size):
        records.append(
            {
                "label": 1.0,
                **pair_features(positives.iloc[index], positives.iloc[index], item_frequency),
            }
        )
        records.append(
            {
                "label": 0.0,
                **pair_features(positives.iloc[index], negatives.iloc[index], item_frequency),
            }
        )
    features = pd.DataFrame.from_records(records)

    comparison_rows = []
    for column in features.columns.drop("label"):
        positive_values = features.loc[features["label"] == 1, column]
        negative_values = features.loc[features["label"] == 0, column]
        comparison_rows.append(
            {
                "feature": column,
                "positive_mean": positive_values.mean(),
                "negative_mean": negative_values.mean(),
                "difference": positive_values.mean() - negative_values.mean(),
                "label_correlation": features[[column, "label"]]
                .corr()
                .iloc[0, 1],
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values(
        "label_correlation", ascending=False
    )
    comparison.to_csv(args.output_dir / "feature_comparison.csv", index=False)

    train_query_norm = train["search_query"].map(normalize)
    benchmark_query_norm = benchmark_queries["search_query"].map(normalize)
    benchmark_item_ids = set(benchmark_items["item_id"])
    reusable_history = train.loc[train["item_id"].isin(benchmark_item_ids), ["item_id"]].copy()
    reusable_history["query_norm"] = train_query_norm[train["item_id"].isin(benchmark_item_ids)]
    history_count = reusable_history.groupby("query_norm")["item_id"].nunique()
    available_history = benchmark_query_norm.map(history_count).fillna(0).astype(int)

    category_zero_items = train.loc[
        train["search_category"].eq(0), "item_category_id"
    ].value_counts()
    report = {
        "sample_size_per_class": sample_size,
        "negative_sampling": "uniform unique item within search_category; category 114 fallback",
        "best_features_by_label_correlation": comparison.head(10).to_dict("records"),
        "benchmark_history": {
            "seen_query_text_count": int(
                benchmark_query_norm.isin(set(train_query_norm)).sum()
            ),
            "queries_with_at_least_one_historical_item_in_benchmark_corpus": int(
                available_history.gt(0).sum()
            ),
            "historical_candidate_count_quantiles": quantiles(available_history),
        },
        "category_zero": {
            "benchmark_query_count": int(
                benchmark_queries["search_category"].eq(0).sum()
            ),
            "train_query_row_count": int(train["search_category"].eq(0).sum()),
            "positive_item_category_counts": {
                str(key): int(value) for key, value in category_zero_items.items()
            },
        },
        "validation_protocol": {
            "cold": "hold out all rows sharing normalized search_query",
            "warm": "hold out one full query signature while retaining other signatures of the same normalized text",
            "primary_reporting": "report cold and warm Recall@50 separately and a 63/37 benchmark-mixture estimate",
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(comparison.to_string(index=False))
    print(f"Wrote reports to {args.output_dir}")


if __name__ == "__main__":
    main()
