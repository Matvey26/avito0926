"""Сохраняет метрики и компактный отчёт эксперимента 0008.

Сырые cold/warm-метрики дополняются общей смесью 63/37, после чего
конфигурации ранжируются совместно и отдельно для BM25 и TF-IDF.
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
    DOCUMENT_VARIANTS,
    MAX_DF,
    MAX_FEATURES,
    MIN_DF,
    MODELS,
    NORMALIZATIONS,
    TOP_K,
    WORD_NGRAMS,
)


CONFIG_COLUMNS = ["model", "normalization", "word_ngram", "document_variant"]


def save_results(
    output_dir: Path,
    result_rows: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    validations: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Сохраняет таблицу метрик и JSON-отчёт и возвращает рейтинг.

    Аргументы:
        output_dir: Каталог результатов эксперимента.
        result_rows: Метрики всех конфигураций и режимов.
        diagnostics: Характеристики построенных словарей и матриц.
        validations: Валидационные таблицы по режимам.

    Возвращает:
        По одной строке на конфигурацию, отсортированные по убыванию смеси
        cold/warm Recall@50.
    """
    metrics = pd.DataFrame(result_rows)
    metrics, comparison = add_mode_mixture(metrics, CONFIG_COLUMNS)
    metrics = metrics.sort_values(CONFIG_COLUMNS + ["mode"], kind="stable")
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    ranked = comparison.sort_values(
        "benchmark_mix_63_cold_37_warm",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    best_by_model = {
        model: ranked.loc[ranked["model"].eq(model)].iloc[0].to_dict()
        for model in MODELS
    }
    report = {
        "grid": {
            "models": list(MODELS),
            "normalizations": list(NORMALIZATIONS),
            "word_ngrams": {key: list(value) for key, value in WORD_NGRAMS.items()},
            "document_variants": list(DOCUMENT_VARIANTS),
            "configurations": len(ranked),
        },
        "shared_vocabulary": {
            "min_df": MIN_DF,
            "max_df": MAX_DF,
            "max_features": MAX_FEATURES,
        },
        "bm25": {"k1": BM25_K1, "b": BM25_B},
        "validation": {
            "cold_queries": len(validations["cold"]),
            "warm_queries": len(validations["warm"]),
            "top_k": TOP_K,
            "category_filter": True,
            "location_bonus": False,
            "character_features": False,
            "fallback": False,
            "mixture": {"cold": 0.63, "warm": 0.37},
        },
        "libraries": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "snowballstemmer",
                "pymorphy3",
            )
        },
        "best_config": ranked.iloc[0].to_dict(),
        "best_by_model": best_by_model,
        "top_configs": ranked.head(20).to_dict("records"),
        "indexes": diagnostics,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ranked
