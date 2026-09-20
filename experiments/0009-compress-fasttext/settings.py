"""Хранит фиксированные параметры dense-retrieval эксперимента 0009.

Единственная перебираемая методологическая ось — состав документа. Модель,
токенизация, mean pooling, cosine similarity, фильтр категории и протокол
валидации остаются одинаковыми во всех конфигурациях.
"""

from __future__ import annotations

import argparse
from pathlib import Path


SEED = "20260918"
VALIDATION_SIZE = 2_500
TOP_K = 50
VECTOR_SIZE = 300
QUERY_BATCH_SIZE = 32
TOKEN_CACHE_SIZE = 200_000
DOCUMENT_VARIANTS = ("title", "title_params", "all_text")

MODEL_FILENAME = "geowac_tokens_sg_300_5_2020-100K-20K-100.bin"
MODEL_URL = (
    "https://github.com/avidale/compress-fasttext/releases/download/"
    "gensim-4-draft/geowac_tokens_sg_300_5_2020-100K-20K-100.bin"
)
MODEL_SHA256 = "5fdb4135d96fab4ab99a64ee71ffd5263c850e4f3f64aec6632c168e70a6cee9"
DEFAULT_MODEL_PATH = Path("assets/models/compress_fasttext") / MODEL_FILENAME
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
    "item_description_raw",
    "item_category_id",
]


def parse_args(description: str) -> argparse.Namespace:
    """Разбирает пути и размер валидации из командной строки.

    Аргументы:
        description: Описание программы для справки ``--help``.

    Возвращает:
        Пространство имён с путями к данным, модели, BM25-метрикам и выходному
        каталогу, а также размером валидации.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=Path("assets/dataset"))
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--bm25-metrics", type=Path, default=DEFAULT_BM25_METRICS)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--validation-size", type=int, default=VALIDATION_SIZE)
    return parser.parse_args()
