"""Сохраняет dense-метрики и сравнение с лучшим BM25 из 0008.

Три состава документа ранжируются по общей смеси cold/warm 63/37. Лучший
dense-вариант сопоставляется с уже зафиксированной BM25-конфигурацией на тех же
валидационных сплитах.
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import pandas as pd

from assets.common_code.metrics import add_mode_mixture
from settings import (
    DOCUMENT_VARIANTS,
    MODEL_FILENAME,
    MODEL_SHA256,
    MODEL_URL,
    TOP_K,
    VECTOR_SIZE,
)


MIXTURE_COLUMN = "benchmark_mix_63_cold_37_warm"


def load_best_bm25(path: Path) -> dict[str, object]:
    """Читает лучшую BM25-конфигурацию из метрик эксперимента 0008.

    Аргументы:
        path: Путь к ``metrics.csv`` эксперимента 0008.

    Возвращает:
        Конфигурацию и её cold, warm и смешанный Recall@50.

    Исключения:
        FileNotFoundError: Если таблица метрик отсутствует.
        ValueError: Если в таблице нет строк BM25.
    """
    metrics = pd.read_csv(path)
    bm25 = metrics.loc[metrics["model"].eq("bm25")]
    if bm25.empty:
        raise ValueError(f"В {path} нет BM25-метрик")
    config_columns = ["normalization", "word_ngram", "document_variant"]
    best_key = (
        bm25.drop_duplicates(config_columns)
        .sort_values(MIXTURE_COLUMN, ascending=False, kind="stable")
        .iloc[0]
    )
    same_config = bm25
    for column in config_columns:
        same_config = same_config.loc[same_config[column].eq(best_key[column])]
    recalls = same_config.set_index("mode")["recall_at_50"]
    return {
        "model": "bm25",
        **{column: best_key[column] for column in config_columns},
        "cold": float(recalls["cold"]),
        "warm": float(recalls["warm"]),
        MIXTURE_COLUMN: float(best_key[MIXTURE_COLUMN]),
    }


def save_results(
    output_dir: Path,
    result_rows: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    validations: dict[str, pd.DataFrame],
    bm25_metrics_path: Path,
    cache_info: object,
) -> pd.DataFrame:
    """Сохраняет метрики, сравнение моделей и JSON-отчёт.

    Аргументы:
        output_dir: Каталог артефактов эксперимента.
        result_rows: Dense-метрики всех составов и режимов.
        diagnostics: Диагностика токенов и пустых векторов по составам.
        validations: Cold/warm-таблицы запросов.
        bm25_metrics_path: Метрики 0008 для контрольного сравнения.
        cache_info: Статистика LRU-кеша word-векторов.

    Возвращает:
        Рейтинг трёх dense-конфигураций по смешанному Recall@50.
    """
    metrics = pd.DataFrame(result_rows)
    metrics, comparison = add_mode_mixture(metrics, ["document_variant"])
    metrics.sort_values(["document_variant", "mode"], kind="stable").to_csv(
        output_dir / "metrics.csv", index=False
    )
    ranked = comparison.sort_values(
        MIXTURE_COLUMN, ascending=False, kind="stable"
    ).reset_index(drop=True)
    best_dense = ranked.iloc[0].to_dict()
    best_dense["model"] = "compress_fasttext_mean"
    best_bm25 = load_best_bm25(bm25_metrics_path)
    comparison_rows = pd.DataFrame([best_dense, best_bm25], dtype=object)
    comparison_rows.to_csv(output_dir / "comparison.csv", index=False)

    report = {
        "model": {
            "library": "compress-fasttext",
            "filename": MODEL_FILENAME,
            "url": MODEL_URL,
            "sha256": MODEL_SHA256,
            "vector_size": VECTOR_SIZE,
            "pooling": "mean raw word vectors, then document L2 normalization",
            "similarity": "exact cosine within target category",
        },
        "grid": {
            "document_variants": list(DOCUMENT_VARIANTS),
            "configurations": len(ranked),
        },
        "validation": {
            "cold_queries": len(validations["cold"]),
            "warm_queries": len(validations["warm"]),
            "top_k": TOP_K,
            "category_filter": True,
            "location_bonus": False,
            "fallback": False,
            "mixture": {"cold": 0.63, "warm": 0.37},
        },
        "libraries": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "compress-fasttext", "gensim")
        },
        "best_dense": best_dense,
        "best_bm25_0008": best_bm25,
        "dense_minus_bm25": best_dense[MIXTURE_COLUMN]
        - best_bm25[MIXTURE_COLUMN],
        "all_dense_configs": ranked.to_dict("records"),
        "diagnostics": diagnostics,
        "token_cache": {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "maxsize": cache_info.maxsize,
            "currsize": cache_info.currsize,
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ranked
