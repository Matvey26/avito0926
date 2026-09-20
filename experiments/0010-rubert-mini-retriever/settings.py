"""Хранит фиксированные параметры dense-retrieval эксперимента 0010.

Эксперимент повторяет протокол 0009 и меняет только способ построения
эмбеддингов: вместо среднего compress-fastText используется готовый
SentenceTransformer ``sergeyzh/rubert-mini-retriever``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


SEED = "20260918"
VALIDATION_SIZE = 2_500
TOP_K = 50
QUERY_BATCH_SIZE = 32
ENCODE_BATCH_SIZE = 128
ENCODE_CHUNK_SIZE = 4_096
VECTOR_SIZE = 312
MAX_SEQUENCE_LENGTH = 512
DOCUMENT_VARIANTS = ("title", "title_params")

MODEL_ID = "sergeyzh/rubert-mini-retriever"
MODEL_REVISION = "4ae6298c749e413b958d64a4b349a907c15a3b54"
MODEL_SHA256 = "45ca37ef2fa6e081e029ec6dc2b0545f2cfe00881c57f31d65319460a236e762"
DEFAULT_MODEL_PATH = Path("assets/models/sergeyzh/rubert-mini-retriever")
DEFAULT_CACHE_DIR = Path("assets/experiments/0010-rubert-mini-retriever")
DEFAULT_BM25_METRICS = Path("experiments/0008-bm25-vs-tfidf/metrics.csv")

QUERY_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]
ITEM_COLUMNS = [
    "item_id",
    "item_title_raw",
    "item_infm_params_text",
    "item_category_id",
]


def parse_args(description: str) -> argparse.Namespace:
    """Разбирает пути и параметры выполнения из командной строки.

    Args:
        description: Описание программы для справки ``--help``.

    Returns:
        Пространство имён с путями к данным, модели, кешу, метрикам BM25 и
        выходному каталогу, размером валидации и выбранным устройством.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=Path("assets/dataset"))
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--bm25-metrics", type=Path, default=DEFAULT_BM25_METRICS)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--validation-size", type=int, default=VALIDATION_SIZE)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Устройство SentenceTransformer; auto выбирает MPS/CUDA/CPU.",
    )
    return parser.parse_args()
