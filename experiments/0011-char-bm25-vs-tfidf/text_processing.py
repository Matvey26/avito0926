"""Нормализует поля и составляет документы эксперимента 0011.

Char-retrieval не использует морфологию: текст приводится к нижнему регистру,
``ё`` заменяется на ``е``, а повторные пробелы сжимаются. Поля добавляются по
одному разу, поэтому скрытого повышения веса заголовка нет.
"""

from __future__ import annotations

import pandas as pd

from settings import DOCUMENT_VARIANTS


def normalize_text(text: object) -> str:
    """Выполняет лёгкую нормализацию одного текстового значения.

    Args:
        text: Произвольный текстовый скаляр или пропуск.

    Returns:
        Нижний регистр с заменой ``ё`` на ``е`` и сжатием пробелов; пропуск
        превращается в пустую строку.
    """
    if text is None or pd.isna(text):
        return ""
    return " ".join(str(text).lower().replace("ё", "е").split())


def normalize_series(values: pd.Series) -> pd.Series:
    """Нормализует pandas Series без изменения порядка и индекса.

    Args:
        values: Исходные текстовые значения.

    Returns:
        Series нормализованных строк с исходным индексом.
    """
    return values.map(normalize_text)


def prepare_item_fields(items: pd.DataFrame) -> dict[str, pd.Series]:
    """Подготавливает три текстовых поля объявлений.

    Args:
        items: Дедуплицированный корпус объявлений.

    Returns:
        Нормализованные заголовки, параметры и описания.
    """
    return {
        "title": normalize_series(items["item_title_raw"]),
        "params": normalize_series(items["item_infm_params_text"]),
        "description": normalize_series(items["item_description_raw"]),
    }


def prepare_query_fields(
    validations: dict[str, pd.DataFrame],
) -> dict[str, dict[str, pd.Series]]:
    """Подготавливает текст и параметры cold/warm-запросов.

    Args:
        validations: Валидационные таблицы по режимам.

    Returns:
        Вложенный словарь ``режим -> поле -> Series``.
    """
    return {
        mode: {
            "query": normalize_series(validation["search_query"]),
            "params": normalize_series(validation["search_infm_params_text"]),
        }
        for mode, validation in validations.items()
    }


def compose_item_documents(fields: dict[str, pd.Series], variant: str) -> pd.Series:
    """Составляет документы объявлений выбранного состава.

    Args:
        fields: Нормализованные поля объявлений.
        variant: ``title``, ``title_params`` или ``all_text``.

    Returns:
        По одной строке на объявление без повторения полей.

    Raises:
        ValueError: Если состав документа неизвестен.
    """
    if variant not in DOCUMENT_VARIANTS:
        raise ValueError(f"Неизвестный состав документа: {variant}")
    if variant == "title":
        return fields["title"]
    documents = fields["title"] + " " + fields["params"]
    if variant == "all_text":
        documents = documents + " " + fields["description"]
    return documents


def compose_query_documents(fields: dict[str, pd.Series], variant: str) -> pd.Series:
    """Составляет запросы под выбранный состав объявления.

    Args:
        fields: Нормализованные поля запросов одного режима.
        variant: ``title``, ``title_params`` или ``all_text``.

    Returns:
        Только ``search_query`` для ``title``; запрос с параметрами для двух
        остальных составов.

    Raises:
        ValueError: Если состав документа неизвестен.
    """
    if variant not in DOCUMENT_VARIANTS:
        raise ValueError(f"Неизвестный состав документа: {variant}")
    if variant == "title":
        return fields["query"]
    return fields["query"] + " " + fields["params"]
