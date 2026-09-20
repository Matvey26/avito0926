"""Хранит фиксированные параметры char-retrieval эксперимента 0011.

Эксперимент сравнивает TF-IDF и Okapi BM25 на общей матрице частот символьных
4-грамм. Единственная перебираемая ось помимо модели — состав документа.
"""

from __future__ import annotations

import argparse
from pathlib import Path


SEED = "20260918"
VALIDATION_SIZE = 2_500
TOP_K = 50
MIN_DF = 2
MAX_DF = 1.0
MAX_FEATURES = 150_000
CHAR_ANALYZER = "char_wb"
CHAR_NGRAM = (4, 4)
BM25_K1 = 1.5
BM25_B = 0.75

MODELS = ("tfidf", "bm25")
DOCUMENT_VARIANTS = ("title", "title_params", "all_text")

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

    Args:
        description: Описание программы для справки ``--help``.

    Returns:
        Пространство имён с каталогами данных и результатов, а также размером
        каждого валидационного сплита.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=Path("assets/dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--validation-size", type=int, default=VALIDATION_SIZE)
    return parser.parse_args()
