#!/usr/bin/env python3
"""Validate an existing answer.csv against the benchmark inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run import validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("assets/dataset"))
    parser.add_argument("--answer", type=Path, default=Path("answer.csv"))
    args = parser.parse_args()

    queries = pd.read_parquet(
        args.data_dir / "benchmark_queries.parquet", columns=["query_id"]
    )
    items = pd.read_parquet(
        args.data_dir / "benchmark_items.parquet", columns=["item_id"]
    )
    answer = pd.read_csv(args.answer, dtype=str, keep_default_na=False)
    validate_submission(answer, queries, set(items["item_id"]))
    lengths = answer["answer"].str.split().str.len()
    print(
        f"OK: {len(answer)} rows, {lengths.min()}..{lengths.max()} candidates per query"
    )


if __name__ == "__main__":
    main()
