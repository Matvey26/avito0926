#!/usr/bin/env python3
"""Запускает чистое сравнение BM25 и word TF-IDF в эксперименте 0008.

Для каждой нормализации, состава документа и диапазона n-грамм строится один
общий словарь CountVectorizer. На его матрице по очереди оцениваются TF-IDF и
Okapi BM25 без char-признаков, бонуса локации и fallback-кандидатов.
"""

from __future__ import annotations

import gc
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assets.common_code.validation import (
    build_validation_splits,
    relevance_ids_to_rows,
)
from indexing import evaluate_models
from reporting import save_results
from settings import (
    DOCUMENT_VARIANTS,
    ITEM_COLUMNS,
    MAX_DF,
    MAX_FEATURES,
    MIN_DF,
    NORMALIZATIONS,
    QUERY_COLUMNS,
    SEED,
    WORD_NGRAMS,
    parse_args,
)
from text_processing import (
    TokenNormalizer,
    compose_item_documents,
    compose_query_documents,
    normalize_text,
    prepare_item_fields,
    prepare_query_fields,
)


def main() -> None:
    """Выполняет полный перебор, сохраняет метрики и печатает лидеров.

    Возвращает:
        Ничего.

    Исключения:
        FileNotFoundError: Если отсутствует ``train.parquet``.
        ValueError: Если CountVectorizer не может построить словарь.
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

    result_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for normalization in NORMALIZATIONS:
        print(f"\nНормализация: {normalization}", flush=True)
        preprocessor = TokenNormalizer(normalization)
        item_fields = prepare_item_fields(items, preprocessor)
        query_fields = prepare_query_fields(validations, preprocessor)

        for variant in DOCUMENT_VARIANTS:
            print(f"  Состав документа: {variant}", flush=True)
            documents = compose_item_documents(item_fields, variant)
            query_documents = {
                mode: compose_query_documents(fields, variant)
                for mode, fields in query_fields.items()
            }

            for word_name, ngram_range in WORD_NGRAMS.items():
                print(f"    Индекс: {word_name}", flush=True)
                started = time.monotonic()
                vectorizer = CountVectorizer(
                    lowercase=False,
                    ngram_range=ngram_range,
                    min_df=MIN_DF,
                    max_df=MAX_DF,
                    max_features=MAX_FEATURES,
                    dtype=np.float32,
                )
                counts = vectorizer.fit_transform(documents).tocsr()
                query_counts = {
                    mode: vectorizer.transform(values).tocsr()
                    for mode, values in query_documents.items()
                }
                indexing_seconds = time.monotonic() - started

                rows, model_diagnostics = evaluate_models(
                    counts,
                    query_counts,
                    validations,
                    relevance_rows,
                    item_categories,
                    item_ids,
                )
                config = {
                    "normalization": normalization,
                    "word_ngram": word_name,
                    "document_variant": variant,
                }
                result_rows.extend({**config, **row} for row in rows)
                diagnostics.append(
                    {
                        **config,
                        "ngram_range": list(ngram_range),
                        "vocabulary_size": len(vectorizer.vocabulary_),
                        "matrix_shape": list(counts.shape),
                        "matrix_nnz": int(counts.nnz),
                        "count_indexing_seconds": indexing_seconds,
                        "normalization_cache_size": len(preprocessor.cache),
                        **model_diagnostics,
                    }
                )
                del vectorizer, counts, query_counts
                gc.collect()
            del documents, query_documents
            gc.collect()
        del item_fields, query_fields, preprocessor
        gc.collect()

    ranked = save_results(
        args.output_dir,
        result_rows,
        diagnostics,
        validations,
    )
    print("\nЛучшие конфигурации:", flush=True)
    print(ranked.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
