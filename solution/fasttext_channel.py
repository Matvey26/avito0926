"""Строит dense retrieval по compress-fastText GeoWAC.

Каждый заголовок объявления и текст запроса представляются L2-нормализованной
суммой исходных 300-мерных word-векторов, как в эксперименте 0009.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
import re
import time
from typing import Any

import compress_fasttext
import numpy as np
import pandas as pd

from retrieval import CandidateBatch, retrieve_dense
from settings import (
    FASTTEXT_MODEL_SHA256,
    FASTTEXT_TOKEN_CACHE_SIZE,
    FASTTEXT_VECTOR_SIZE,
)
from text_processing import normalize_text


TOKEN_RE = re.compile(r"(?u)\b\w{2,}\b")


def file_sha256(path: Path) -> str:
    """Вычисляет SHA-256 файла модели.

    Args:
        path: Путь к локальному файлу.

    Returns:
        Шестнадцатеричную контрольную сумму.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path: Path) -> Any:
    """Проверяет и загружает compress-fastText GeoWAC.

    Args:
        path: Путь к модели.

    Returns:
        Загруженные keyed vectors.

    Raises:
        FileNotFoundError: Если модель не скачана.
        RuntimeError: Если контрольная сумма или размерность неверна.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {path}. Запустите solution/download_model.py"
        )
    actual_hash = file_sha256(path)
    if actual_hash != FASTTEXT_MODEL_SHA256:
        raise RuntimeError(f"Неверный SHA-256 fastText-модели: {actual_hash}")
    model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(str(path))
    if model.vector_size != FASTTEXT_VECTOR_SIZE:
        raise RuntimeError("Неожиданная размерность fastText-модели")
    return model


class TextEmbedder:
    """Кодирует тексты суммой word-векторов с кешированием токенов."""

    def __init__(self, model: Any) -> None:
        """Сохраняет загруженную модель.

        Args:
            model: Объект compress-fastText keyed vectors.
        """
        self.model = model

    @lru_cache(maxsize=FASTTEXT_TOKEN_CACHE_SIZE)
    def token_vector(self, token: str) -> np.ndarray:
        """Возвращает кешированный вектор токена.

        Args:
            token: Нормализованный токен.

        Returns:
            Вектор float32 размерности 300.
        """
        return np.asarray(self.model[token], dtype=np.float32)

    def encode(self, values: pd.Series, label: str) -> np.ndarray:
        """Кодирует и L2-нормализует последовательность текстов.

        Args:
            values: Тексты в рабочем порядке.
            label: Подпись этапа в прогрессе.

        Returns:
            Матрицу единичных векторов.
        """
        started = time.monotonic()
        matrix = np.zeros((len(values), FASTTEXT_VECTOR_SIZE), dtype=np.float32)
        for row_number, value in enumerate(values):
            for token in TOKEN_RE.findall(normalize_text(value)):
                matrix[row_number] += self.token_vector(token)
            if (row_number + 1) % 25_000 == 0 or row_number + 1 == len(values):
                print(f"  {label}: {row_number + 1:,}/{len(values):,}", flush=True)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        np.divide(matrix, norms, out=matrix, where=norms > 0)
        print(f"  {label}: {time.monotonic() - started:.1f} с", flush=True)
        return matrix


def build_fasttext_channel(
    items: pd.DataFrame,
    queries: pd.DataFrame,
    item_categories: np.ndarray,
    item_ids: np.ndarray,
    model_path: Path,
) -> CandidateBatch:
    """Строит точный cosine top-500 по заголовкам объявлений.

    Args:
        items: Benchmark-корпус объявлений.
        queries: Benchmark-запросы.
        item_categories: Категории строк корпуса.
        item_ids: Идентификаторы строк корпуса.
        model_path: Путь к проверенной fastText-модели.

    Returns:
        Top-500 fastText внутри категории.
    """
    print("Подготовка fastText...", flush=True)
    embedder = TextEmbedder(load_model(model_path))
    item_vectors = embedder.encode(items["item_title_raw"], "fastText: заголовки")
    query_vectors = embedder.encode(queries["search_query"], "fastText: запросы")
    result = retrieve_dense(
        item_vectors,
        query_vectors,
        queries,
        item_categories,
        item_ids,
    )
    print("  fastText: кандидаты готовы", flush=True)
    return result
