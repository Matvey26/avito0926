"""Строит fastText title-retrieval из эксперимента 0009.

Тексты кодируются средним исходных 300-мерных word-векторов GeoWAC. После
L2-нормализации выполняется точный cosine top-500 внутри категории.
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

from checkpointing import CandidateStore
from retrieval import retrieve_dense
from settings import (
    FASTTEXT_MODEL_SHA256,
    FASTTEXT_TOKEN_CACHE_SIZE,
    FASTTEXT_VECTOR_SIZE,
)
from text_processing import normalize_text


TOKEN_RE = re.compile(r"(?u)\b\w{2,}\b")


def file_sha256(path: Path) -> str:
    """Вычисляет SHA-256 локального файла модели.

    Args:
        path: Путь к модели.

    Returns:
        Шестнадцатеричный SHA-256.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path: Path) -> Any:
    """Проверяет и загружает compress-fastText GeoWAC.

    Args:
        path: Локальный путь модели 0009.

    Returns:
        Объект keyed vectors размерности 300.

    Raises:
        FileNotFoundError: Если модель не подготовлена.
        RuntimeError: Если SHA-256 или размерность неверны.
    """
    if not path.exists():
        raise FileNotFoundError(f"Модель не найдена: {path}")
    actual_hash = file_sha256(path)
    if actual_hash != FASTTEXT_MODEL_SHA256:
        raise RuntimeError(f"Неверный SHA-256 fastText-модели: {actual_hash}")
    model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(str(path))
    if model.vector_size != FASTTEXT_VECTOR_SIZE:
        raise RuntimeError("Неожиданная размерность fastText-модели")
    return model


class TextEmbedder:
    """Кодирует строки суммой word-векторов с LRU-кешем токенов."""

    def __init__(self, model: Any) -> None:
        """Сохраняет модель.

        Args:
            model: Загруженные compress-fastText keyed vectors.
        """
        self.model = model

    @lru_cache(maxsize=FASTTEXT_TOKEN_CACHE_SIZE)
    def token_vector(self, token: str) -> np.ndarray:
        """Возвращает кешированный вектор токена.

        Args:
            token: Нормализованный токен.

        Returns:
            Вектор ``float32`` размерности 300.
        """
        return np.asarray(self.model[token], dtype=np.float32)

    def encode(self, values: pd.Series, label: str) -> np.ndarray:
        """Кодирует Series и L2-нормализует суммы токенов.

        Args:
            values: Тексты в рабочем порядке.
            label: Название для сообщений прогресса.

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
        print(f"  {label}: готово за {time.monotonic() - started:.1f} с", flush=True)
        return matrix


def ensure_fasttext(
    items: pd.DataFrame,
    validations: dict[str, pd.DataFrame],
    item_categories: np.ndarray,
    item_ids: np.ndarray,
    model_path: Path,
    store: CandidateStore,
) -> None:
    """Строит отсутствующий fastText title checkpoint.

    Args:
        items: Корпус объявлений.
        validations: Cold/warm-запросы.
        item_categories: Категории корпуса.
        item_ids: Item ID корпуса.
        model_path: Путь к модели 0009.
        store: Хранилище кандидатов.

    Returns:
        Ничего.
    """
    modes = tuple(validations)
    if store.complete("fasttext", modes):
        print("fasttext: checkpoint готов", flush=True)
        return
    print("Подготовка fastText...", flush=True)
    started = time.monotonic()
    embedder = TextEmbedder(load_model(model_path))
    item_embeddings = embedder.encode(items["item_title_raw"], "fastText: заголовки")
    query_embeddings = {
        mode: embedder.encode(frame["search_query"], f"fastText: {mode}")
        for mode, frame in validations.items()
    }
    batches = retrieve_dense(
        item_embeddings,
        query_embeddings,
        validations,
        item_categories,
        item_ids,
    )
    cache_info = embedder.token_vector.cache_info()
    store.save(
        "fasttext",
        batches,
        {
            "document_variant": "title",
            "pooling": "mean raw word vectors and L2 normalization",
            "model_sha256": FASTTEXT_MODEL_SHA256,
            "vector_size": FASTTEXT_VECTOR_SIZE,
            "seconds": time.monotonic() - started,
            "token_cache": {
                "hits": cache_info.hits,
                "misses": cache_info.misses,
                "maxsize": cache_info.maxsize,
                "currsize": cache_info.currsize,
            },
        },
    )
