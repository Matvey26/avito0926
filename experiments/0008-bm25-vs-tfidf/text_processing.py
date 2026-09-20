"""Нормализует текст и составляет документы эксперимента 0008.

Все сравниваемые модели получают одинаковые заранее обработанные поля. Режим
нормализации меняет только морфологию токенов, а вариант документа — только
набор конкатенируемых полей без дополнительного повторения и скрытых весов.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable

import pandas as pd
import pymorphy3
import snowballstemmer

from settings import DOCUMENT_VARIANTS, NORMALIZATIONS


TOKEN_RE = re.compile(r"(?u)\b\w{2,}\b")
RUSSIAN_TOKEN_RE = re.compile(r"^[а-я]+$")


def normalize_text(text: object) -> str:
    """Применяет общую лёгкую нормализацию текста.

    Аргументы:
        text: Произвольный текстовый скаляр или пропущенное значение.

    Возвращает:
        Нижний регистр с заменой ``ё`` на ``е`` и сжатием пробелов; пропуски
        превращаются в пустую строку.
    """
    if text is None or pd.isna(text):
        return ""
    return " ".join(str(text).lower().replace("ё", "е").split())


class TokenNormalizer:
    """Применяет выбранную морфологию и кеширует результаты по токенам.

    Атрибуты:
        mode: Режим ``plain``, ``russian_snowball`` или ``russian_lemma``.
        cache: Соответствие исходных токенов их нормализованным формам.
        stemmer: Русский Snowball-стеммер в соответствующем режиме.
        morph: Анализатор Pymorphy3 в режиме лемматизации.
    """

    def __init__(self, mode: str) -> None:
        """Создаёт нормализатор выбранного режима.

        Аргументы:
            mode: Один из режимов ``settings.NORMALIZATIONS``.

        Исключения:
            ValueError: Если передан неизвестный режим.
        """
        if mode not in NORMALIZATIONS:
            raise ValueError(f"Неизвестная нормализация: {mode}")
        self.mode = mode
        self.cache: dict[str, str] = {}
        self.stemmer = (
            snowballstemmer.stemmer("russian")
            if mode == "russian_snowball"
            else None
        )
        self.morph = pymorphy3.MorphAnalyzer() if mode == "russian_lemma" else None

    def _normalize_unseen(self, tokens: list[str]) -> None:
        """Добавляет нормализованные формы новых токенов в кеш.

        Аргументы:
            tokens: Уникальные токены, которых ещё нет в ``cache``.

        Возвращает:
            Ничего; метод изменяет ``cache`` на месте.
        """
        if self.mode == "russian_snowball":
            assert self.stemmer is not None
            self.cache.update(zip(tokens, self.stemmer.stemWords(tokens)))
            return
        if self.mode == "russian_lemma":
            assert self.morph is not None
            for token in tokens:
                if RUSSIAN_TOKEN_RE.fullmatch(token):
                    lemma = self.morph.parse(token)[0].normal_form.replace("ё", "е")
                else:
                    lemma = token
                self.cache[token] = lemma
            return
        self.cache.update((token, token) for token in tokens)

    def __call__(self, text: object) -> str:
        """Токенизирует и нормализует одно текстовое значение.

        Аргументы:
            text: Значение, принимаемое :func:`normalize_text`.

        Возвращает:
            Нормализованные токены длиной не менее двух символов через пробел.
        """
        tokens = TOKEN_RE.findall(normalize_text(text))
        unseen = list(dict.fromkeys(token for token in tokens if token not in self.cache))
        if unseen:
            self._normalize_unseen(unseen)
        return " ".join(self.cache[token] for token in tokens)


def preprocess_series(
    values: pd.Series,
    preprocessor: Callable[[object], str],
    label: str,
) -> pd.Series:
    """Обрабатывает Series порциями и печатает прогресс.

    Аргументы:
        values: Исходные текстовые значения.
        preprocessor: Функция обработки одного значения.
        label: Название поля для сообщений о прогрессе.

    Возвращает:
        Series нормализованных строк с исходным индексом.
    """
    started = time.monotonic()
    result: list[str] = []
    chunk_size = 25_000
    total = len(values)
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        result.extend(preprocessor(value) for value in values.iloc[start:stop])
        print(f"  {label}: {stop:,}/{total:,}", flush=True)
    print(f"  {label}: готово за {time.monotonic() - started:.1f} с", flush=True)
    return pd.Series(result, index=values.index, dtype="object")


def prepare_item_fields(
    items: pd.DataFrame,
    preprocessor: TokenNormalizer,
) -> dict[str, pd.Series]:
    """Нормализует текстовые поля корпуса объявлений.

    Аргументы:
        items: Корпус с заголовками, параметрами и описаниями.
        preprocessor: Нормализатор выбранного режима.

    Возвращает:
        Словарь нормализованных Series с ключами ``title``, ``params`` и
        ``description``.
    """
    return {
        "title": preprocess_series(items["item_title_raw"], preprocessor, "заголовки"),
        "params": preprocess_series(
            items["item_infm_params_text"], preprocessor, "параметры объявлений"
        ),
        "description": preprocess_series(
            items["item_description_raw"], preprocessor, "описания"
        ),
    }


def prepare_query_fields(
    validations: dict[str, pd.DataFrame],
    preprocessor: TokenNormalizer,
) -> dict[str, dict[str, pd.Series]]:
    """Нормализует текст и фильтры всех валидационных запросов.

    Аргументы:
        validations: Валидационные таблицы по режимам.
        preprocessor: Тот же нормализатор, что использован для объявлений.

    Возвращает:
        Вложенный словарь ``режим -> поле -> Series`` для ``query`` и
        ``params``.
    """
    result: dict[str, dict[str, pd.Series]] = {}
    for mode, validation in validations.items():
        result[mode] = {
            "query": preprocess_series(
                validation["search_query"], preprocessor, f"{mode}: запросы"
            ),
            "params": preprocess_series(
                validation["search_infm_params_text"],
                preprocessor,
                f"{mode}: фильтры",
            ),
        }
    return result


def compose_item_documents(fields: dict[str, pd.Series], variant: str) -> pd.Series:
    """Составляет документы объявлений выбранного варианта.

    Аргументы:
        fields: Нормализованные поля из :func:`prepare_item_fields`.
        variant: ``title``, ``title_params`` или ``all_text``.

    Возвращает:
        По одному документу на объявление без повторения отдельных полей.

    Исключения:
        ValueError: Если вариант документа неизвестен.
    """
    if variant not in DOCUMENT_VARIANTS:
        raise ValueError(f"Неизвестный состав документа: {variant}")
    if variant == "title":
        return fields["title"]
    result = fields["title"] + " " + fields["params"]
    if variant == "all_text":
        result = result + " " + fields["description"]
    return result


def compose_query_documents(fields: dict[str, pd.Series], variant: str) -> pd.Series:
    """Составляет документы запросов под вариант документа объявления.

    Аргументы:
        fields: Нормализованные поля одного валидационного режима.
        variant: ``title``, ``title_params`` или ``all_text``.

    Возвращает:
        Только текст запроса для ``title``; текст запроса и фильтров для двух
        остальных вариантов.

    Исключения:
        ValueError: Если вариант документа неизвестен.
    """
    if variant not in DOCUMENT_VARIANTS:
        raise ValueError(f"Неизвестный состав документа: {variant}")
    if variant == "title":
        return fields["query"]
    return fields["query"] + " " + fields["params"]
