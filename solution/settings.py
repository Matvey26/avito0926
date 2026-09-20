"""Хранит зафиксированную конфигурацию итогового решения.

Значения перенесены из победившей стратегии эксперимента 0012 и не
подбираются на benchmark-запросах.
"""

from pathlib import Path


TOP_K = 50
CANDIDATE_POOL = 500
LOCATION_BONUS = 0.5

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
FASTTEXT_MODEL_URL = (
    "https://github.com/avidale/compress-fasttext/releases/download/"
    "gensim-4-draft/geowac_tokens_sg_300_5_2020-100K-20K-100.bin"
)
FASTTEXT_MODEL_SHA256 = (
    "5fdb4135d96fab4ab99a64ee71ffd5263c850e4f3f64aec6632c168e70a6cee9"
)
DEFAULT_MODEL_PATH = (
    Path("assets/models/compress_fasttext") / FASTTEXT_MODEL_FILENAME
)

QUERY_COLUMNS = [
    "query_id",
    "search_query",
    "search_location_id",
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


def target_category(value: object) -> int:
    """Определяет категорию корпуса для запроса.

    Args:
        value: Значение ``search_category``.

    Returns:
        Категорию 114 для нулевой категории, иначе исходное целое значение.
    """
    category = int(value)
    return 114 if category == 0 else category
