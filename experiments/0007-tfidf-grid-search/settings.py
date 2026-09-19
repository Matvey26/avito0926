"""Fixed experiment configuration and command-line arguments."""

from __future__ import annotations

import argparse
from pathlib import Path


SEED = "20260918"
BOOTSTRAP_SEED = 20260919
BOOTSTRAP_RUNS = 10
VALIDATION_SIZE = 2_500
TOP_K = 50
CANDIDATE_POOL = 500
CHAR_WEIGHTS = [round(step / 10, 1) for step in range(11)]
LOCATION_BOOSTS = sorted(
    {round(step / 10, 1) for step in range(11)} | {0.56}
)
WORD_NGRAMS = {
    "unigram": (1, 1),
    "unigram_bigram": (1, 2),
    "unigram_to_trigram": (1, 3),
}
CHAR_NGRAMS = {
    f"{minimum}_to_{maximum}": (minimum, maximum)
    for minimum in range(2, 7)
    for maximum in range(minimum, 7)
    if minimum < maximum or minimum == 4
}
NORMALIZATIONS = ("plain", "russian_snowball", "russian_lemma")
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
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=Path("assets/dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--validation-size", type=int, default=VALIDATION_SIZE)
    return parser.parse_args()
