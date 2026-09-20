#!/usr/bin/env python3
"""Сравнивает fusion 0012 с word (1,2) и char (3,4)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CODE = PROJECT_ROOT / "experiments/0012-three-retriever-fusion"
for path in (PROJECT_ROOT, BASE_CODE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Имя отделено от settings.py эксперимента 0012, который импортируют его модули.
import importlib.util

spec = importlib.util.spec_from_file_location(
    "settings_0013", Path(__file__).with_name("settings.py")
)
if spec is None or spec.loader is None:
    raise RuntimeError("Не удалось загрузить настройки 0013")
settings_0013 = importlib.util.module_from_spec(spec)
sys.modules["settings_0013"] = settings_0013
spec.loader.exec_module(settings_0013)

from assets.common_code.metrics import add_mode_mixture
from assets.common_code.validation import build_validation_splits, relevance_ids_to_rows
from checkpointing import CandidateStore, data_fingerprint
from fusion import evaluate_fusion, evaluate_retrievers
from settings import ITEM_COLUMNS, QUERY_COLUMNS, SEED, VALIDATION_SIZE
from text_processing import normalize_text
from variants import ensure_char_variant, ensure_word_variant


def parse_args() -> argparse.Namespace:
    """Разбирает пути данных, кешей и результатов.

    Returns:
        Аргументы командной строки.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("assets/dataset"))
    parser.add_argument("--cache-dir", type=Path, default=settings_0013.DEFAULT_CACHE_DIR)
    parser.add_argument("--base-cache-dir", type=Path, default=settings_0013.BASE_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    return parser.parse_args()


def main() -> None:
    """Строит два изменённых канала и сравнивает fusion с 0012.

    Raises:
        RuntimeError: Если отсутствует fastText checkpoint эксперимента 0012.
        OSError: Если данные или результаты недоступны.
    """
    args = parse_args()
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
        VALIDATION_SIZE,
        normalize_text,
        query_columns=QUERY_COLUMNS,
        seed=SEED,
    )
    relevance_rows = {
        mode: relevance_ids_to_rows(values, item_row_by_id)
        for mode, values in relevance_ids.items()
    }
    modes = tuple(validations)
    fingerprint = data_fingerprint(items, validations)
    store = CandidateStore(args.cache_dir, fingerprint)
    base_store = CandidateStore(args.base_cache_dir, fingerprint)

    ensure_word_variant(items, validations, item_categories, item_ids, store)
    ensure_char_variant(items, validations, item_categories, item_ids, store)
    if not base_store.complete("fasttext", modes):
        raise RuntimeError("Сначала нужен fastText checkpoint эксперимента 0012")

    batches = {
        "word_bm25": store.load("word_bm25_1_2", modes),
        "fasttext": base_store.load("fasttext", modes),
        "char_bm25": store.load("char_bm25_3_4", modes),
    }
    retriever_metrics = pd.DataFrame(
        evaluate_retrievers(batches, validations, relevance_rows)
    )
    fusion_metrics = pd.DataFrame(
        evaluate_fusion(
            batches,
            validations,
            relevance_rows,
            item_locations,
            item_ids,
        )
    )
    fusion_metrics, ranked = add_mode_mixture(fusion_metrics, ["strategy"])
    fusion_metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    retriever_metrics.to_csv(args.output_dir / "retriever_metrics.csv", index=False)

    base = pd.read_csv(settings_0013.BASE_METRICS_PATH)
    base_mix = float(
        base.loc[
            base["strategy"].eq("mean_minmax"),
            "benchmark_mix_63_cold_37_warm",
        ].iloc[0]
    )
    variant_mix = float(
        ranked.loc[
            ranked["strategy"].eq("mean_minmax"),
            "benchmark_mix_63_cold_37_warm",
        ].iloc[0]
    )
    comparison = pd.DataFrame(
        [
            {"configuration": "0012_word_1_1_char_4_4", "mix_recall_at_50": base_mix},
            {"configuration": "0013_word_1_2_char_3_4", "mix_recall_at_50": variant_mix},
        ]
    )
    comparison["delta_vs_0012"] = comparison["mix_recall_at_50"] - base_mix
    comparison.to_csv(args.output_dir / "comparison.csv", index=False)
    report = {
        "word_ngram": list(settings_0013.WORD_NGRAM),
        "char_ngram": list(settings_0013.CHAR_NGRAM),
        "location_bonus": 0.5,
        "base_mix_recall_at_50": base_mix,
        "variant_mix_recall_at_50": variant_mix,
        "delta_vs_0012": variant_mix - base_mix,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
