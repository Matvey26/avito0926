"""Сохраняет метрики и попарное сравнение моделей 0011.

Cold/warm-метрики дополняются смесью 63/37. Отдельная таблица показывает
разницу BM25 и TF-IDF при строго одинаковом составе документа.
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import pandas as pd

from assets.common_code.metrics import add_mode_mixture
from settings import (
    BM25_B,
    BM25_K1,
    CHAR_ANALYZER,
    CHAR_NGRAM,
    DOCUMENT_VARIANTS,
    MAX_DF,
    MAX_FEATURES,
    MIN_DF,
    MODELS,
    TOP_K,
)


MIXTURE_COLUMN = "benchmark_mix_63_cold_37_warm"
CONFIG_COLUMNS = ["model", "document_variant"]


def build_pairwise(comparison: pd.DataFrame) -> pd.DataFrame:
    """Строит попарную таблицу BM25 против TF-IDF.

    Args:
        comparison: По одной строке на модель и состав документа.

    Returns:
        Cold, warm и смешанные оценки обеих моделей с разницей смеси.
    """
    pairwise = comparison.pivot(
        index="document_variant", columns="model", values=["cold", "warm", MIXTURE_COLUMN]
    )
    pairwise.columns = [f"{metric}_{model}" for metric, model in pairwise.columns]
    pairwise = pairwise.reset_index()
    pairwise["bm25_minus_tfidf"] = (
        pairwise[f"{MIXTURE_COLUMN}_bm25"] - pairwise[f"{MIXTURE_COLUMN}_tfidf"]
    )
    order = {variant: number for number, variant in enumerate(DOCUMENT_VARIANTS)}
    return pairwise.sort_values(
        "document_variant", key=lambda values: values.map(order), kind="stable"
    ).reset_index(drop=True)


def save_results(
    output_dir: Path,
    result_rows: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    validations: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Сохраняет CSV-метрики, comparison и JSON-отчёт.

    Args:
        output_dir: Каталог небольших артефактов эксперимента.
        result_rows: Метрики моделей, составов и сплитов.
        diagnostics: Характеристики построенных char-индексов.
        validations: Cold/warm-таблицы запросов.

    Returns:
        Рейтинг всех конфигураций по смешанному Recall@50.
    """
    metrics = pd.DataFrame(result_rows)
    metrics, comparison = add_mode_mixture(metrics, CONFIG_COLUMNS)
    metrics.sort_values(CONFIG_COLUMNS + ["mode"], kind="stable").to_csv(
        output_dir / "metrics.csv", index=False
    )
    ranked = comparison.sort_values(
        MIXTURE_COLUMN, ascending=False, kind="stable"
    ).reset_index(drop=True)
    pairwise = build_pairwise(comparison)
    pairwise.to_csv(output_dir / "comparison.csv", index=False)
    best_by_model = {
        model: ranked.loc[ranked["model"].eq(model)].iloc[0].to_dict()
        for model in MODELS
    }
    report = {
        "grid": {
            "models": list(MODELS),
            "char_analyzer": CHAR_ANALYZER,
            "char_ngram": list(CHAR_NGRAM),
            "document_variants": list(DOCUMENT_VARIANTS),
            "configurations": len(ranked),
        },
        "shared_vocabulary": {
            "min_df": MIN_DF,
            "max_df": MAX_DF,
            "max_features": MAX_FEATURES,
        },
        "tfidf": {"sublinear_tf": True, "smooth_idf": True, "norm": "l2"},
        "bm25": {"k1": BM25_K1, "b": BM25_B},
        "validation": {
            "cold_queries": len(validations["cold"]),
            "warm_queries": len(validations["warm"]),
            "top_k": TOP_K,
            "category_filter": True,
            "word_features": False,
            "location_bonus": False,
            "fallback": False,
            "mixture": {"cold": 0.63, "warm": 0.37},
        },
        "libraries": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn")
        },
        "best_config": ranked.iloc[0].to_dict(),
        "best_by_model": best_by_model,
        "pairwise": pairwise.to_dict("records"),
        "indexes": diagnostics,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ranked
