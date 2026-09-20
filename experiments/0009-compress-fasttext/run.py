#!/usr/bin/env python3
"""Запускает dense retrieval на русских compress-fastText-эмбеддингах.

Эксперимент один раз кодирует текстовые поля объявлений и запросов, после чего
сравнивает три состава документа при фиксированных mean pooling, cosine
similarity и фильтре категории. Лучший вариант сопоставляется с BM25 из 0008.
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

from assets.common_code.validation import (
    build_validation_splits,
    relevance_ids_to_rows,
)
from embeddings import (
    TextEmbedder,
    compose_embeddings,
    load_model,
    normalize_text,
    prepare_item_fields,
    prepare_query_fields,
)
from reporting import save_results
from retrieval import evaluate_dense_splits
from settings import (
    DOCUMENT_VARIANTS,
    ITEM_COLUMNS,
    QUERY_COLUMNS,
    SEED,
    parse_args,
)


def main() -> None:
    """Кодирует данные, оценивает три состава и сохраняет результаты.

    Возвращает:
        Ничего.

    Исключения:
        FileNotFoundError: Если отсутствуют train или локальная модель.
        RuntimeError: Если модель не проходит проверку SHA-256 или размерности.
        OSError: Если данные нельзя прочитать или результаты записать.
    """
    args = parse_args(__doc__)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(
        args.data_dir / "train.parquet",
        columns=QUERY_COLUMNS + ITEM_COLUMNS,
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

    print(f"Загрузка модели: {args.model_path}", flush=True)
    model = load_model(args.model_path)
    embedder = TextEmbedder(model)
    print("Кодирование полей объявлений...", flush=True)
    item_fields = prepare_item_fields(items, embedder)
    print("Кодирование полей запросов...", flush=True)
    query_fields = prepare_query_fields(validations, embedder)

    result_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for variant in DOCUMENT_VARIANTS:
        print(f"\nОценка состава: {variant}", flush=True)
        item_embeddings, item_token_counts = compose_embeddings(
            item_fields, variant, query_side=False
        )
        query_embeddings: dict[str, np.ndarray] = {}
        query_token_counts: dict[str, np.ndarray] = {}
        for mode, fields in query_fields.items():
            query_embeddings[mode], query_token_counts[mode] = compose_embeddings(
                fields, variant, query_side=True
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
                "mean_item_tokens": float(item_token_counts.mean()),
                "zero_item_embeddings": int(np.count_nonzero(item_token_counts == 0)),
                "mean_query_tokens": {
                    mode: float(values.mean())
                    for mode, values in query_token_counts.items()
                },
                "zero_query_embeddings": {
                    mode: int(np.count_nonzero(values == 0))
                    for mode, values in query_token_counts.items()
                },
            }
        )
        del item_embeddings, item_token_counts, query_embeddings, query_token_counts
        gc.collect()

    ranked = save_results(
        args.output_dir,
        result_rows,
        diagnostics,
        validations,
        args.bm25_metrics,
        embedder.token_vector.cache_info(),
    )
    print("\nDense-конфигурации:", flush=True)
    print(ranked.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
