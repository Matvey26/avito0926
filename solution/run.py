#!/usr/bin/env python3
"""Создаёт ``answer.csv`` победившим mean-minmax fusion.

Пайплайн независимо строит word BM25, fastText и char BM25, извлекает по 500
кандидатов каждого канала внутри категории, объединяет их оценки и добавляет
фиксированный бонус точного совпадения локации.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd

from fasttext_channel import build_fasttext_channel
from fusion import fuse_all
from settings import DEFAULT_MODEL_PATH, ITEM_COLUMNS, QUERY_COLUMNS
from sparse_channels import build_char_channel, build_word_channel
from submission import build_submission, validate_submission


def parse_args() -> argparse.Namespace:
    """Разбирает пути к данным, модели и итоговому CSV.

    Returns:
        Аргументы командной строки с объектами :class:`pathlib.Path`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("assets/dataset"))
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=Path("answer.csv"))
    return parser.parse_args()


def main() -> None:
    """Запускает три retriever-канала, fusion и запись submission.

    Raises:
        FileNotFoundError: Если отсутствуют данные или fastText-модель.
        RuntimeError: Если fastText-модель не совпадает с зафиксированной.
        ValueError: Если созданный submission нарушает формат задачи.
        OSError: Если входные или выходной файлы недоступны.
    """
    args = parse_args()
    started = time.monotonic()
    queries = pd.read_parquet(
        args.data_dir / "benchmark_queries.parquet", columns=QUERY_COLUMNS
    )
    items = pd.read_parquet(
        args.data_dir / "benchmark_items.parquet", columns=ITEM_COLUMNS
    )
    item_ids = items["item_id"].astype(str).to_numpy()
    item_categories = items["item_category_id"].to_numpy()
    item_locations = items["item_location_id"].to_numpy()
    print(
        f"Данные: {len(queries):,} запросов, {len(items):,} объявлений",
        flush=True,
    )

    word = build_word_channel(items, queries, item_categories, item_ids)
    char = build_char_channel(items, queries, item_categories, item_ids)
    fasttext = build_fasttext_channel(
        items, queries, item_categories, item_ids, args.model_path
    )
    predictions = fuse_all(
        [word, fasttext, char], queries, item_locations, item_ids
    )
    answer = build_submission(queries, predictions, item_ids)
    validate_submission(answer, queries, set(item_ids))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    answer.to_csv(args.output, index=False)
    print(
        f"Готово: {args.output} ({time.monotonic() - started:.1f} с)",
        flush=True,
    )


if __name__ == "__main__":
    main()
