"""Готовит тексты word BM25, char BM25 и fastText.

Word-канал повторяет лемматизацию 0008. Char- и fastText-каналы используют
лёгкую нормализацию; составы полей соответствуют победителям 0011 и 0009.
"""

from __future__ import annotations

import re
import time

import pandas as pd
import pymorphy3


TOKEN_RE = re.compile(r"(?u)\b\w{2,}\b")
RUSSIAN_TOKEN_RE = re.compile(r"^[а-я]+$")


def normalize_text(text: object) -> str:
    """Применяет общую лёгкую нормализацию текста.

    Args:
        text: Произвольный текстовый скаляр или пропуск.

    Returns:
        Нижний регистр с заменой ``ё`` на ``е`` и сжатием пробелов.
    """
    if text is None or pd.isna(text):
        return ""
    return " ".join(str(text).lower().replace("ё", "е").split())


class RussianLemmatizer:
    """Лемматизирует русские токены и кеширует результаты."""

    def __init__(self) -> None:
        """Создаёт Pymorphy3-анализатор и пустой кеш."""
        self.morph = pymorphy3.MorphAnalyzer()
        self.cache: dict[str, str] = {}

    def __call__(self, text: object) -> str:
        """Токенизирует и лемматизирует одно значение.

        Args:
            text: Значение, поддерживаемое :func:`normalize_text`.

        Returns:
            Токены длиной от двух символов через пробел.
        """
        tokens = TOKEN_RE.findall(normalize_text(text))
        for token in dict.fromkeys(tokens):
            if token in self.cache:
                continue
            if RUSSIAN_TOKEN_RE.fullmatch(token):
                lemma = self.morph.parse(token)[0].normal_form.replace("ё", "е")
            else:
                lemma = token
            self.cache[token] = lemma
        return " ".join(self.cache[token] for token in tokens)


def preprocess_series(
    values: pd.Series, processor: object, label: str
) -> pd.Series:
    """Обрабатывает Series порциями с сообщениями о прогрессе.

    Args:
        values: Исходные значения.
        processor: Вызываемый обработчик одного значения.
        label: Название поля для прогресса.

    Returns:
        Обработанную Series в исходном порядке.
    """
    started = time.monotonic()
    result: list[str] = []
    for start in range(0, len(values), 25_000):
        stop = min(start + 25_000, len(values))
        result.extend(processor(value) for value in values.iloc[start:stop])
        print(f"  {label}: {stop:,}/{len(values):,}", flush=True)
    print(f"  {label}: готово за {time.monotonic() - started:.1f} с", flush=True)
    return pd.Series(result, index=values.index, dtype="object")


def compose_all_text(
    title: pd.Series, params: pd.Series, description: pd.Series
) -> pd.Series:
    """Конкатенирует три поля ровно по одному разу.

    Args:
        title: Подготовленные заголовки.
        params: Подготовленные параметры.
        description: Подготовленные описания.

    Returns:
        Документы ``title + params + description``.
    """
    return title + " " + params + " " + description


def prepare_word_documents(
    items: pd.DataFrame, validations: dict[str, pd.DataFrame]
) -> tuple[pd.Series, dict[str, pd.Series], int]:
    """Строит лемматизированные документы word BM25 из 0008.

    Args:
        items: Корпус объявлений.
        validations: Cold/warm-запросы.

    Returns:
        Документы объявлений, запросы по режимам и размер кеша лемм.
    """
    lemmatizer = RussianLemmatizer()
    title = preprocess_series(items["item_title_raw"], lemmatizer, "word: заголовки")
    params = preprocess_series(
        items["item_infm_params_text"], lemmatizer, "word: параметры"
    )
    description = preprocess_series(
        items["item_description_raw"], lemmatizer, "word: описания"
    )
    documents = compose_all_text(title, params, description)
    queries = {}
    for mode, validation in validations.items():
        query = preprocess_series(
            validation["search_query"], lemmatizer, f"word: {mode} запросы"
        )
        query_params = preprocess_series(
            validation["search_infm_params_text"],
            lemmatizer,
            f"word: {mode} параметры",
        )
        queries[mode] = query + " " + query_params
    return documents, queries, len(lemmatizer.cache)


def prepare_char_documents(
    items: pd.DataFrame, validations: dict[str, pd.DataFrame]
) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Строит plain-документы char BM25 из 0011.

    Args:
        items: Корпус объявлений.
        validations: Cold/warm-запросы.

    Returns:
        Все текстовые поля объявлений и запросы с параметрами.
    """
    title = items["item_title_raw"].map(normalize_text)
    params = items["item_infm_params_text"].map(normalize_text)
    description = items["item_description_raw"].map(normalize_text)
    documents = compose_all_text(title, params, description)
    queries = {
        mode: validation["search_query"].map(normalize_text)
        + " "
        + validation["search_infm_params_text"].map(normalize_text)
        for mode, validation in validations.items()
    }
    return documents, queries
