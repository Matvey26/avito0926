"""Сохраняет метрики 0010 и сравнение с лучшим BM25 из 0008.

Два состава документа ранжируются по смеси cold/warm 63/37. Победитель
нейросетевого retrieval сопоставляется с зафиксированным BM25 на тех же
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
    MAX_SEQUENCE_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    TOP_K,
    VECTOR_SIZE,
)


MIXTURE_COLUMN = "benchmark_mix_63_cold_37_warm"


def load_best_bm25(path: Path) -> dict[str, object]:
    """Читает лучшую BM25-конфигурацию из эксперимента 0008.

    Args:
        path: Путь к таблице ``metrics.csv`` эксперимента 0008.

    Returns:
        Конфигурацию и её cold, warm и смешанный Recall@50.

    Raises:
        ValueError: Если в таблице отсутствуют строки BM25.
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
    device: str,
) -> pd.DataFrame:
    """Сохраняет метрики, сравнение моделей и JSON-отчёт.

    Args:
        output_dir: Каталог небольших артефактов эксперимента.
        result_rows: Dense-метрики всех составов и режимов.
        diagnostics: Диагностика кодирования по составам.
        validations: Cold/warm-таблицы запросов.
        bm25_metrics_path: Метрики 0008 для контрольного сравнения.
        device: Фактически использованное устройство PyTorch.

    Returns:
        Рейтинг двух dense-конфигураций по смешанному Recall@50.
    """
    metrics = pd.DataFrame(result_rows)
    metrics, comparison = add_mode_mixture(metrics, ["document_variant"])
    metrics.sort_values(["document_variant", "mode"], kind="stable").to_csv(
        output_dir / "metrics.csv", index=False
    )
    ranked = comparison.sort_values(MIXTURE_COLUMN, ascending=False, kind="stable").reset_index(drop=True)
    best_dense = ranked.iloc[0].to_dict()
    best_dense["model"] = "sergeyzh/rubert-mini-retriever"
    best_bm25 = load_best_bm25(bm25_metrics_path)
    pd.DataFrame([best_dense, best_bm25], dtype=object).to_csv(
        output_dir / "comparison.csv", index=False
    )

    report = {
        "model": {
            "library": "sentence-transformers",
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "model_sha256": MODEL_SHA256,
            "vector_size": VECTOR_SIZE,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "pooling": "native SentenceTransformer pooling and L2 normalization",
            "similarity": "exact cosine within target category",
            "query_prefix": None,
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
        "runtime": {"device": device},
        "libraries": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "torch", "sentence-transformers", "transformers")
        },
        "best_dense": best_dense,
        "best_bm25_0008": best_bm25,
        "dense_minus_bm25": best_dense[MIXTURE_COLUMN] - best_bm25[MIXTURE_COLUMN],
        "all_dense_configs": ranked.to_dict("records"),
        "diagnostics": diagnostics,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ranked
