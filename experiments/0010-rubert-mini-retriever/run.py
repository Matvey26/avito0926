#!/usr/bin/env python3
"""Запускает dense retrieval на ``sergeyzh/rubert-mini-retriever``.

Эксперимент повторяет 0009: оценивает два коротких состава документа на тех же
cold/warm-сплитах, использует точный cosine внутри категории и сравнивает
лучший вариант с BM25 из 0008. Отличается только модель построения векторов.
"""

from __future__ import annotations

import gc
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assets.common_code.validation import build_validation_splits, relevance_ids_to_rows
from embeddings import choose_device, encode_frame, load_model, normalize_text
from reporting import save_results
from retrieval import evaluate_dense_splits
from settings import DOCUMENT_VARIANTS, ITEM_COLUMNS, QUERY_COLUMNS, SEED, parse_args


def cache_path(cache_dir: Path, kind: str, variant: str, size: int | None = None) -> Path:
    """Формирует устойчивое имя файла кеша эмбеддингов.

    Args:
        cache_dir: Общий каталог тяжёлых артефактов 0010.
        kind: ``items``, ``cold_queries`` или ``warm_queries``.
        variant: Состав текста.
        size: Размер валидации для query-кеша; для корпуса не задаётся.

    Returns:
        Путь NPY-файла внутри ``cache_dir``.
    """
    suffix = f"_{size}" if size is not None else ""
    return cache_dir / f"{kind}_{variant}{suffix}.npy"


def main() -> None:
    """Кодирует данные, оценивает два состава и сохраняет результаты.

    Returns:
        Ничего.

    Raises:
        FileNotFoundError: Если отсутствуют train или локальная модель.
        RuntimeError: Если модель либо устройство не соответствуют настройкам.
        OSError: Если данные или результаты нельзя прочитать и записать.
    """
    args = parse_args(__doc__)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(
        args.data_dir / "train.parquet", columns=QUERY_COLUMNS + ITEM_COLUMNS
    )
    items = train[ITEM_COLUMNS].drop_duplicates("item_id").reset_index(drop=True)
    item_ids = items["item_id"].to_numpy()
    item_categories = items["item_category_id"].to_numpy()
    item_row_by_id = dict(zip(item_ids, items.index))

    validations, relevance_ids = build_validation_splits(
        train,
        args.validation_size,
        normalize_text,
        query_columns=QUERY_COLUMNS,
        seed=SEED,
    )
    relevance_rows = {
        mode: relevance_ids_to_rows(mode_relevance, item_row_by_id)
        for mode, mode_relevance in relevance_ids.items()
    }

    device = choose_device(args.device)
    print(f"Загрузка модели: {args.model_path}; устройство: {device}", flush=True)
    model = load_model(args.model_path, device)
    result_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    for variant in DOCUMENT_VARIANTS:
        print(f"\nКодирование и оценка состава: {variant}", flush=True)
        item_embeddings, item_diagnostics = encode_frame(
            model,
            items,
            variant,
            query_side=False,
            output=cache_path(args.cache_dir, "items", variant),
        )
        query_embeddings: dict[str, np.ndarray] = {}
        query_diagnostics: dict[str, dict[str, object]] = {}
        for mode, validation in validations.items():
            query_embeddings[mode], query_diagnostics[mode] = encode_frame(
                model,
                validation,
                variant,
                query_side=True,
                output=cache_path(
                    args.cache_dir, f"{mode}_queries", variant, args.validation_size
                ),
            )

        rows = evaluate_dense_splits(
            item_embeddings,
            query_embeddings,
            validations,
            relevance_rows,
            item_categories,
            item_ids,
        )
        result_rows.extend({"document_variant": variant, **row} for row in rows)
        diagnostics.append(
            {
                "document_variant": variant,
                "items": item_diagnostics,
                "queries": query_diagnostics,
            }
        )
        del item_embeddings, query_embeddings
        gc.collect()

    ranked = save_results(
        args.output_dir,
        result_rows,
        diagnostics,
        validations,
        args.bm25_metrics,
        device,
    )
    print("\nDense-конфигурации:", flush=True)
    print(ranked.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
