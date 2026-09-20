"""Хранит фиксированные параметры fusion-эксперимента 0012.

Три retrieval-канала повторяют лучшие конфигурации 0008, 0009 и 0011.
Перебирается только стратегия fusion; размер пула и location bonus фиксированы.
"""

from __future__ import annotations

import argparse
from pathlib import Path


SEED = "20260918"
VALIDATION_SIZE = 2_500
TOP_K = 50
CANDIDATE_POOL = 500
LOCATION_BONUS = 0.5
RRF_K = 60
FUSION_STRATEGIES = ("rrf", "mean_minmax")
RETRIEVERS = ("word_bm25", "fasttext", "char_bm25")

BM25_K1 = 1.5
BM25_B = 0.75
WORD_MIN_DF = 2
WORD_MAX_DF = 0.99
WORD_MAX_FEATURES = 200_000
CHAR_MIN_DF = 2
CHAR_MAX_DF = 1.0
CHAR_MAX_FEATURES = 150_000
CHAR_NGRAM = (4, 4)

FASTTEXT_VECTOR_SIZE = 300
FASTTEXT_QUERY_BATCH_SIZE = 32
FASTTEXT_TOKEN_CACHE_SIZE = 200_000
FASTTEXT_MODEL_FILENAME = "geowac_tokens_sg_300_5_2020-100K-20K-100.bin"
FASTTEXT_MODEL_SHA256 = (
    "5fdb4135d96fab4ab99a64ee71ffd5263c850e4f3f64aec6632c168e70a6cee9"
)
DEFAULT_FASTTEXT_MODEL = (
    Path("assets/models/compress_fasttext") / FASTTEXT_MODEL_FILENAME
)
DEFAULT_CACHE_DIR = Path("assets/experiments/0012-three-retriever-fusion")

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
    "item_location_id",
]


def parse_args(description: str) -> argparse.Namespace:
    """Разбирает пути и размер валидации из командной строки.

    Args:
        description: Описание программы для справки ``--help``.

    Returns:
        Пространство имён с каталогами данных, кеша и результатов, путём
        fastText-модели и размером валидационного сплита.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=Path("assets/dataset"))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_FASTTEXT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--validation-size", type=int, default=VALIDATION_SIZE)
    return parser.parse_args()
