"""Строит dense-представления текстовых полей с compress-fastText.

Каждое поле кодируется суммой исходных word-векторов модели. При составлении
документа суммы выбранных полей складываются, что эквивалентно mean pooling
всех его токенов после итоговой L2-нормализации.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import re
import time
from typing import Any

import compress_fasttext
import numpy as np
import pandas as pd

from settings import DOCUMENT_VARIANTS, MODEL_SHA256, TOKEN_CACHE_SIZE, VECTOR_SIZE


TOKEN_RE = re.compile(r"(?u)\b\w{2,}\b")


@dataclass
class EmbeddedField:
    """Хранит суммы word-векторов и числа токенов одного поля.

    Атрибуты:
        sums: Матрица сумм размерности ``число строк × VECTOR_SIZE``.
        token_counts: Число токенов в каждой строке.
    """

    sums: np.ndarray
    token_counts: np.ndarray


def normalize_text(text: object) -> str:
    """Приводит текст к формату токенизации модели.

    Аргументы:
        text: Произвольный текстовый скаляр или пропуск.

    Возвращает:
        Нижний регистр с заменой ``ё`` на ``е`` и сжатием пробелов; пропуски
        превращаются в пустую строку.
    """
    if text is None or pd.isna(text):
        return ""
    return " ".join(str(text).lower().replace("ё", "е").split())


def file_sha256(path: Path) -> str:
    """Вычисляет SHA-256 локального файла модели.

    Аргументы:
        path: Путь к файлу.

    Возвращает:
        Шестнадцатеричный SHA-256 нижнего регистра.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path: Path) -> Any:
    """Проверяет контрольную сумму и загружает compress-fastText-модель.

    Аргументы:
        path: Локальный путь модели.

    Возвращает:
        Экземпляр ``CompressedFastTextKeyedVectors`` размерности 300.

    Исключения:
        FileNotFoundError: Если модель не скачана.
        RuntimeError: Если SHA-256 или размерность модели неверны.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {path}. Запустите download_model.py"
        )
    actual_hash = file_sha256(path)
    if actual_hash != MODEL_SHA256:
        raise RuntimeError(f"Неверный SHA-256 модели: {actual_hash}")
    model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(str(path))
    if model.vector_size != VECTOR_SIZE:
        raise RuntimeError(
            f"Ожидалась размерность {VECTOR_SIZE}, получена {model.vector_size}"
        )
    return model


class TextEmbedder:
    """Кодирует тексты суммой word-векторов с ограниченным кешем токенов.

    Атрибуты:
        model: Загруженная compress-fastText-модель.
    """

    def __init__(self, model: Any) -> None:
        """Сохраняет модель для последующего кодирования.

        Аргументы:
            model: Совместимый с Gensim объект fastText keyed vectors.
        """
        self.model = model

    @lru_cache(maxsize=TOKEN_CACHE_SIZE)
    def token_vector(self, token: str) -> np.ndarray:
        """Возвращает и кеширует исходный вектор одного токена.

        Аргументы:
            token: Нормализованный токен.

        Возвращает:
            Вектор ``float32`` размерности ``VECTOR_SIZE``. FastText строит
            OOV-векторы из символьных n-грамм.
        """
        return np.asarray(self.model[token], dtype=np.float32)

    def encode_series(self, values: pd.Series, label: str) -> EmbeddedField:
        """Кодирует текстовую Series и показывает прогресс.

        Аргументы:
            values: Тексты с произвольным индексом.
            label: Название поля для сообщений о прогрессе.

        Возвращает:
            Суммы word-векторов и числа токенов в исходном порядке.
        """
        started = time.monotonic()
        sums = np.zeros((len(values), VECTOR_SIZE), dtype=np.float32)
        counts = np.zeros(len(values), dtype=np.int32)
        progress_step = 25_000
        for row_number, value in enumerate(values):
            tokens = TOKEN_RE.findall(normalize_text(value))
            target = sums[row_number]
            for token in tokens:
                target += self.token_vector(token)
            counts[row_number] = len(tokens)
            if (row_number + 1) % progress_step == 0 or row_number + 1 == len(values):
                print(f"  {label}: {row_number + 1:,}/{len(values):,}", flush=True)
        print(f"  {label}: готово за {time.monotonic() - started:.1f} с", flush=True)
        return EmbeddedField(sums=sums, token_counts=counts)


def prepare_item_fields(
    items: pd.DataFrame,
    embedder: TextEmbedder,
) -> dict[str, EmbeddedField]:
    """Кодирует три текстовых поля корпуса объявлений.

    Аргументы:
        items: Корпус с заголовком, параметрами и описанием.
        embedder: Настроенный кодировщик fastText.

    Возвращает:
        Поля ``title``, ``params`` и ``description`` в dense-виде.
    """
    return {
        "title": embedder.encode_series(items["item_title_raw"], "заголовки"),
        "params": embedder.encode_series(
            items["item_infm_params_text"], "параметры объявлений"
        ),
        "description": embedder.encode_series(
            items["item_description_raw"], "описания"
        ),
    }


def prepare_query_fields(
    validations: dict[str, pd.DataFrame],
    embedder: TextEmbedder,
) -> dict[str, dict[str, EmbeddedField]]:
    """Кодирует текст и параметры cold/warm-запросов.

    Аргументы:
        validations: Валидационные таблицы по режимам.
        embedder: Тот же кодировщик, что применялся к объявлениям.

    Возвращает:
        Вложенный словарь ``режим -> поле -> EmbeddedField``.
    """
    result: dict[str, dict[str, EmbeddedField]] = {}
    for mode, validation in validations.items():
        result[mode] = {
            "query": embedder.encode_series(
                validation["search_query"], f"{mode}: запросы"
            ),
            "params": embedder.encode_series(
                validation["search_infm_params_text"], f"{mode}: фильтры"
            ),
        }
    return result


def compose_embeddings(
    fields: dict[str, EmbeddedField],
    variant: str,
    query_side: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Собирает и L2-нормализует embedding выбранного состава.

    Аргументы:
        fields: Dense-поля объявления или запроса.
        variant: ``title``, ``title_params`` или ``all_text``.
        query_side: Использовать ключи ``query``/``params`` вместо полей item.

    Возвращает:
        Матрицу единичных векторов и числа токенов каждого документа.

    Исключения:
        ValueError: Если вариант документа неизвестен.
    """
    if variant not in DOCUMENT_VARIANTS:
        raise ValueError(f"Неизвестный состав документа: {variant}")
    first = "query" if query_side else "title"
    sums = fields[first].sums.copy()
    counts = fields[first].token_counts.copy()
    if variant != "title":
        sums += fields["params"].sums
        counts += fields["params"].token_counts
    if variant == "all_text" and not query_side:
        sums += fields["description"].sums
        counts += fields["description"].token_counts
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    np.divide(sums, norms, out=sums, where=norms > 0)
    return sums, counts
