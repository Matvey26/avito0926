"""Хранит фиксированную сетку и аргументы эксперимента 0008.

В модуле собраны все параметры, влияющие на сравнение BM25 и TF-IDF: размеры
валидации и выдачи, ограничения словаря, диапазоны word n-грамм, варианты
нормализации, составы документов и фиксированные коэффициенты BM25.
"""

from __future__ import annotations

import argparse
from pathlib import Path


SEED = "20260918"
VALIDATION_SIZE = 2_500
TOP_K = 50
MIN_DF = 2
MAX_DF = 0.99
MAX_FEATURES = 200_000
BM25_K1 = 1.5
BM25_B = 0.75

MODELS = ("tfidf", "bm25")
NORMALIZATIONS = ("plain", "russian_snowball", "russian_lemma")
WORD_NGRAMS = {
    "unigram": (1, 1),
    "unigram_bigram": (1, 2),
    "unigram_to_trigram": (1, 3),
}
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

    Аргументы:
        description: Описание программы для справки ``--help``.

    Возвращает:
        Пространство имён с путями ``data_dir`` и ``output_dir`` и целым
        ``validation_size``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=Path("assets/dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--validation-size", type=int, default=VALIDATION_SIZE)
    return parser.parse_args()
