"""Сохраняет fusion-метрики и подробный отчёт 0012."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import pandas as pd

from assets.common_code.metrics import add_mode_mixture
from checkpointing import CandidateStore
from settings import (
    CANDIDATE_POOL,
    FUSION_STRATEGIES,
    LOCATION_BONUS,
    RETRIEVERS,
    RRF_K,
    TOP_K,
)


MIXTURE_COLUMN = "benchmark_mix_63_cold_37_warm"


def add_retriever_mixtures(metrics: pd.DataFrame) -> pd.DataFrame:
    """Добавляет смеси 63/37 для Recall@50 и полного пула.

    Args:
        metrics: Диагностика retrieval в длинном формате.

    Returns:
        Исходные строки с двумя колонками смешанных метрик.
    """
    result = metrics.copy()
    for metric in ("recall_at_50", "recall_at_pool"):
        wide = metrics.pivot(index="retriever", columns="mode", values=metric)
        mixture = 0.63 * wide["cold"] + 0.37 * wide["warm"]
        result = result.merge(
            mixture.rename(f"benchmark_mix_{metric}").reset_index(),
            on="retriever",
            how="left",
        )
    return result


def save_results(
    output_dir: Path,
    fusion_rows: list[dict[str, object]],
    retriever_rows: list[dict[str, object]],
    validations: dict[str, pd.DataFrame],
    store: CandidateStore,
) -> pd.DataFrame:
    """Сохраняет CSV и JSON-артефакты эксперимента.

    Args:
        output_dir: Каталог результатов.
        fusion_rows: Recall@50 стратегий по режимам.
        retriever_rows: Recall@50/500 отдельных каналов.
        validations: Cold/warm-запросы.
        store: Хранилище с диагностикой построения каналов.

    Returns:
        Рейтинг fusion-стратегий по смешанному Recall@50.
    """
    metrics = pd.DataFrame(fusion_rows)
    metrics, ranked = add_mode_mixture(metrics, ["strategy"])
    metrics.sort_values(["strategy", "mode"], kind="stable").to_csv(
        output_dir / "metrics.csv", index=False
    )
    ranked = ranked.sort_values(
        MIXTURE_COLUMN, ascending=False, kind="stable"
    ).reset_index(drop=True)
    retriever_metrics = add_retriever_mixtures(pd.DataFrame(retriever_rows))
    retriever_metrics.sort_values(["retriever", "mode"], kind="stable").to_csv(
        output_dir / "retriever_metrics.csv", index=False
    )
    retriever_records = (
        retriever_metrics.astype(object)
        .where(pd.notna(retriever_metrics), None)
        .to_dict("records")
    )
    report = {
        "retrievers": {
            "word_bm25": "0008: russian_lemma, unigram, all_text",
            "fasttext": "0009: GeoWAC mean pooling, title",
            "char_bm25": "0011: char_wb (4,4), all_text",
        },
        "fusion": {
            "strategies": list(FUSION_STRATEGIES),
            "candidate_pool_per_retriever": CANDIDATE_POOL,
            "top_k": TOP_K,
            "rrf_k": RRF_K,
            "missing_minmax_score": 0.0,
            "location_bonus": LOCATION_BONUS,
            "location_bonus_stage": "after fusion score normalization",
        },
        "validation": {
            "cold_queries": len(validations["cold"]),
            "warm_queries": len(validations["warm"]),
            "mixture": {"cold": 0.63, "warm": 0.37},
        },
        "libraries": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "pymorphy3",
                "compress-fasttext",
            )
        },
        "best_fusion": ranked.iloc[0].to_dict(),
        "all_fusions": ranked.to_dict("records"),
        "retriever_metrics": retriever_records,
        "retriever_builds": {
            name: store.diagnostics(name) for name in RETRIEVERS
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ranked
