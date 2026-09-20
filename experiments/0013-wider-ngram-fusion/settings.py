"""Хранит параметры варианта fusion с расширенными n-граммами."""

from pathlib import Path


WORD_NGRAM = (1, 2)
CHAR_NGRAM = (3, 4)
WORD_MAX_FEATURES = 200_000
CHAR_MAX_FEATURES = 150_000
MIN_DF = 2
WORD_MAX_DF = 0.99
CHAR_MAX_DF = 1.0

DEFAULT_CACHE_DIR = Path("assets/experiments/0013-wider-ngram-fusion")
BASE_CACHE_DIR = Path("assets/experiments/0012-three-retriever-fusion")
BASE_METRICS_PATH = Path("experiments/0012-three-retriever-fusion/metrics.csv")
