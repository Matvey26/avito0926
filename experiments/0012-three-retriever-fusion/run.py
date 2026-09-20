#!/usr/bin/env python3
"""Запускает fusion word BM25, fastText и char BM25.

Каждый retriever сохраняет top-500 с исходными score. Затем на их объединении
сравниваются RRF и среднее min-max нормализованных score с location bonus 0.5.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assets.common_code.validation import build_validation_splits, relevance_ids_to_rows
from checkpointing import CandidateStore, data_fingerprint
from fasttext_retriever import ensure_fasttext
from fusion import evaluate_fusion, evaluate_retrievers
from reporting import save_results
from settings import ITEM_COLUMNS, QUERY_COLUMNS, RETRIEVERS, SEED, parse_args
from sparse_retrievers import ensure_char_bm25, ensure_word_bm25
from text_processing import normalize_text


def main() -> None:
    """Строит retrieval-checkpoint, оценивает fusion и сохраняет отчёт.

    Returns:
        Ничего.

    Raises:
        FileNotFoundError: Если отсутствуют данные или fastText-модель.
        RuntimeError: Если модель или checkpoint некорректны.
        OSError: Если артефакты нельзя прочитать или записать.
    """
    args = parse_args(__doc__)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(
        args.data_dir / "train.parquet", columns=QUERY_COLUMNS + ITEM_COLUMNS
    )
    items = train[ITEM_COLUMNS].drop_duplicates("item_id").reset_index(drop=True)
    item_ids = items["item_id"].to_numpy()
    item_categories = items["item_category_id"].to_numpy()
    item_locations = items["item_location_id"].to_numpy()
    item_row_by_id = dict(zip(item_ids, items.index))
    validations, relevance_ids = build_validation_splits(
        train,
        args.validation_size,
        normalize_text,
        query_columns=QUERY_COLUMNS,
        seed=SEED,
    )
    relevance_rows = {
        mode: relevance_ids_to_rows(values, item_row_by_id)
        for mode, values in relevance_ids.items()
    }
    modes = tuple(validations)
    store = CandidateStore(args.cache_dir, data_fingerprint(items, validations))

    ensure_word_bm25(items, validations, item_categories, item_ids, store)
    ensure_char_bm25(items, validations, item_categories, item_ids, store)
    ensure_fasttext(
        items,
        validations,
        item_categories,
        item_ids,
        args.model_path,
        store,
    )
    batches = {name: store.load(name, modes) for name in RETRIEVERS}
    retriever_rows = evaluate_retrievers(batches, validations, relevance_rows)
    fusion_rows = evaluate_fusion(
        batches,
        validations,
        relevance_rows,
        item_locations,
        item_ids,
    )
    ranked = save_results(
        args.output_dir,
        fusion_rows,
        retriever_rows,
        validations,
        store,
    )
    print("\nFusion-стратегии:", flush=True)
    print(ranked.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
