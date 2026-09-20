#!/usr/bin/env python3
"""Проверяет готовый ``answer.csv`` по данным benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from submission import validate_submission


def main() -> None:
    """Читает идентификаторы benchmark и валидирует submission.

    Raises:
        FileNotFoundError: Если отсутствуют данные или проверяемый CSV.
        ValueError: Если CSV нарушает формат задачи.
        OSError: Если один из файлов недоступен для чтения.
    """
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
    validate_submission(answer, queries, set(items["item_id"].astype(str)))
    lengths = answer["answer"].str.split().str.len()
    print(
        f"OK: {len(answer)} строк, {lengths.min()}..{lengths.max()} "
        "кандидатов на запрос"
    )


if __name__ == "__main__":
    main()
