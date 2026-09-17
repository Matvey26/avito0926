#!/usr/bin/env python3
"""Build a compact, reproducible structural report for all task datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq


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
    "item_category_id",
    "item_microcat_id",
    "item_price",
    "item_rating",
    "item_rating_reviews_count",
    "item_location_id",
    "item_latitude",
    "item_longitude",
    "item_is_phone_hidden",
    "item_is_message_forbidden",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("assets/dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    return parser.parse_args()


def scalar(value: Any) -> Any:
    """Convert pandas/numpy scalar values to JSON-compatible Python values."""
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def column_summary(series: pd.Series) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dtype": str(series.dtype),
        "null_count": int(series.isna().sum()),
        "unique_count": int(series.nunique(dropna=True)),
    }
    if pd.api.types.is_bool_dtype(series):
        result["value_counts"] = {
            str(key): int(value)
            for key, value in series.value_counts(dropna=False).items()
        }
    elif pd.api.types.is_numeric_dtype(series):
        values = series.dropna()
        result["quantiles"] = {
            str(q): scalar(value)
            for q, value in values.quantile([0, 0.5, 0.9, 0.99, 1]).items()
        }
    else:
        lengths = series.fillna("").astype(str).str.len()
        result["length_quantiles"] = {
            str(q): float(value)
            for q, value in lengths.quantile([0, 0.5, 0.9, 0.99, 1]).items()
        }
    return result


def description_length_summary(path: Path) -> dict[str, float | int]:
    """Read only the large description column and avoid a full-frame materialization."""
    descriptions = pq.read_table(path, columns=["item_description_raw"])[
        "item_description_raw"
    ]
    lengths = pc.utf8_length(pc.fill_null(descriptions, "")).to_numpy()
    series = pd.Series(lengths)
    return {
        "null_count": int(descriptions.null_count),
        "unique_count": int(pc.count_distinct(descriptions).as_py()),
        "length_min": int(series.min()),
        "length_median": float(series.quantile(0.5)),
        "length_p90": float(series.quantile(0.9)),
        "length_p99": float(series.quantile(0.99)),
        "length_max": int(series.max()),
    }


def frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "columns": {column: column_summary(frame[column]) for column in frame.columns},
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.data_dir / "train.parquet"
    query_path = args.data_dir / "benchmark_queries.parquet"
    item_path = args.data_dir / "benchmark_items.parquet"

    # Descriptions are deliberately loaded separately because they dominate memory.
    train = pd.read_parquet(train_path, columns=QUERY_COLUMNS + ITEM_COLUMNS)
    queries = pd.read_parquet(query_path)
    items = pd.read_parquet(item_path, columns=ITEM_COLUMNS)

    train_item_ids = set(train["item_id"])
    benchmark_item_ids = set(items["item_id"])
    item_overlap = train_item_ids & benchmark_item_ids
    train_texts = set(train["search_query"].str.lower().str.strip())
    benchmark_normalized = queries["search_query"].str.lower().str.strip()

    positives_per_signature = train.groupby(QUERY_COLUMNS, dropna=False)[
        "item_id"
    ].nunique()
    rows_per_signature = train.groupby(QUERY_COLUMNS, dropna=False).size()

    report = {
        "datasets": {
            "train": frame_summary(train),
            "benchmark_queries": frame_summary(queries),
            "benchmark_items": frame_summary(items),
        },
        "description_lengths": {
            "train": description_length_summary(train_path),
            "benchmark_items": description_length_summary(item_path),
        },
        "integrity": {
            "train_unique_query_signatures": int(
                len(train[QUERY_COLUMNS].drop_duplicates())
            ),
            "train_unique_query_texts": int(train["search_query"].nunique()),
            "train_unique_item_ids": int(train["item_id"].nunique()),
            "benchmark_unique_item_ids": int(items["item_id"].nunique()),
            "benchmark_unique_query_ids": int(queries["query_id"].nunique()),
            "benchmark_item_ids_valid": bool(
                items["item_id"].str.fullmatch(r"[0-9a-f]{16}").all()
            ),
            "benchmark_query_ids_length_16": bool(
                queries["query_id"].str.len().eq(16).all()
            ),
        },
        "overlap": {
            "train_benchmark_item_count": len(item_overlap),
            "benchmark_item_fraction_seen_in_train": len(item_overlap)
            / len(benchmark_item_ids),
            "train_item_fraction_in_benchmark": len(item_overlap) / len(train_item_ids),
            "benchmark_rows_with_seen_normalized_query_text": int(
                benchmark_normalized.isin(train_texts).sum()
            ),
            "benchmark_fraction_with_seen_normalized_query_text": float(
                benchmark_normalized.isin(train_texts).mean()
            ),
        },
        "positive_pair_rules": {
            "category_exact_match_rate": float(
                (train["search_category"] == train["item_category_id"]).mean()
            ),
            "location_exact_match_rate": float(
                (train["search_location_id"] == train["item_location_id"]).mean()
            ),
        },
        "group_statistics": {
            "positives_per_query_signature": {
                str(q): float(value)
                for q, value in positives_per_signature.quantile(
                    [0, 0.5, 0.9, 0.95, 0.99, 1]
                ).items()
            },
            "rows_per_query_signature": {
                str(q): float(value)
                for q, value in rows_per_signature.quantile(
                    [0, 0.5, 0.9, 0.99, 1]
                ).items()
            },
        },
        "top_values": {
            "train_search_category": {
                str(key): int(value)
                for key, value in train["search_category"].value_counts().items()
            },
            "benchmark_search_category": {
                str(key): int(value)
                for key, value in queries["search_category"].value_counts().items()
            },
            "train_delivery_flag": {
                str(key): int(value)
                for key, value in train["search_is_delivery_search"]
                .value_counts()
                .items()
            },
        },
    }

    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
