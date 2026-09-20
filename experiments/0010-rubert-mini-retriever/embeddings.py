"""Строит и кеширует эмбеддинги ``rubert-mini-retriever``.

Тексты составляются из тех же полей, что в эксперименте 0009. Модель кодирует
полный состав документа своим токенизатором и pooling-слоем, после чего вектор
L2-нормализуется для точного cosine retrieval. Крупные матрицы сохраняются в
``assets`` и повторно используются при безопасном перезапуске.
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from download_model import REVISION_MARKER, REQUIRED_FILES
from settings import (
    DOCUMENT_VARIANTS,
    ENCODE_BATCH_SIZE,
    ENCODE_CHUNK_SIZE,
    MAX_SEQUENCE_LENGTH,
    MODEL_REVISION,
    MODEL_SHA256,
    VECTOR_SIZE,
)


def normalize_text(text: object) -> str:
    """Выполняет общую для экспериментов нормализацию текста.

    Args:
        text: Текстовый скаляр или пропуск.

    Returns:
        Строку в нижнем регистре, с ``ё`` заменённой на ``е`` и сжатыми
        пробелами; пропуск превращается в пустую строку.
    """
    if text is None or pd.isna(text):
        return ""
    return " ".join(str(text).lower().replace("ё", "е").split())


def file_sha256(path: Path) -> str:
    """Вычисляет SHA-256 локального файла весов порциями.

    Args:
        path: Путь к ``model.safetensors``.

    Returns:
        Шестнадцатеричный SHA-256 нижнего регистра.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_device(requested: str) -> str:
    """Выбирает доступное устройство для инференса модели.

    Args:
        requested: Явное ``cpu``, ``mps``, ``cuda`` либо ``auto``.

    Returns:
        Название устройства PyTorch.

    Raises:
        RuntimeError: Если явно запрошенное устройство недоступно.
    """
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }
    if requested == "auto":
        return next(name for name in ("mps", "cuda", "cpu") if available[name])
    if not available[requested]:
        raise RuntimeError(f"Устройство {requested!r} недоступно")
    return requested


def load_model(path: Path, device: str) -> SentenceTransformer:
    """Проверяет локальный snapshot и загружает SentenceTransformer offline.

    Args:
        path: Каталог модели, подготовленный ``download_model.py``.
        device: Устройство PyTorch для инференса.

    Returns:
        Готовую модель с размерностью 312 и контекстом 512 токенов.

    Raises:
        FileNotFoundError: Если snapshot не был подготовлен.
        RuntimeError: Если ревизия, размерность или длина контекста неверны.
    """
    marker = path / REVISION_MARKER
    if not marker.exists() or any(not (path / name).exists() for name in REQUIRED_FILES):
        raise FileNotFoundError(f"Модель не подготовлена: {path}")
    actual_revision = marker.read_text(encoding="utf-8").strip()
    if actual_revision != MODEL_REVISION:
        raise RuntimeError(f"Ожидалась ревизия {MODEL_REVISION}, получена {actual_revision}")
    actual_hash = file_sha256(path / "model.safetensors")
    if actual_hash != MODEL_SHA256:
        raise RuntimeError(f"Неверный SHA-256 весов модели: {actual_hash}")
    model = SentenceTransformer(str(path), device=device, local_files_only=True)
    if model.get_sentence_embedding_dimension() != VECTOR_SIZE:
        raise RuntimeError("Неожиданная размерность эмбеддинга модели")
    if model.max_seq_length != MAX_SEQUENCE_LENGTH:
        raise RuntimeError(f"Ожидался контекст {MAX_SEQUENCE_LENGTH}, получен {model.max_seq_length}")
    return model


def compose_texts(frame: pd.DataFrame, variant: str, query_side: bool) -> list[str]:
    """Составляет тексты одного батча из выбранных полей.

    Args:
        frame: Батч запросов или объявлений.
        variant: ``title`` или ``title_params``.
        query_side: Использовать поля запроса вместо полей объявления.

    Returns:
        Нормализованные строки в исходном порядке.

    Raises:
        ValueError: Если указан неизвестный состав документа.
    """
    if variant not in DOCUMENT_VARIANTS:
        raise ValueError(f"Неизвестный состав документа: {variant}")
    if query_side:
        columns = ["search_query"]
        if variant != "title":
            columns.append("search_infm_params_text")
    else:
        columns = ["item_title_raw"]
        if variant != "title":
            columns.append("item_infm_params_text")
    values = frame[columns].itertuples(index=False, name=None)
    return [" ".join(filter(None, (normalize_text(value) for value in row))) for row in values]


def cache_is_valid(path: Path, rows: int) -> bool:
    """Проверяет форму матрицы и метаданные кеша.

    Args:
        path: Путь к NPY-матрице.
        rows: Ожидаемое число строк.

    Returns:
        ``True`` для совместимого завершённого кеша.
    """
    metadata_path = path.with_suffix(".json")
    if not path.exists() or not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {"rows": rows, "columns": VECTOR_SIZE, "model_revision": MODEL_REVISION}
    if any(metadata.get(key) != value for key, value in expected.items()):
        return False
    return np.load(path, mmap_mode="r").shape == (rows, VECTOR_SIZE)


def encode_frame(
    model: SentenceTransformer,
    frame: pd.DataFrame,
    variant: str,
    query_side: bool,
    output: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    """Кодирует DataFrame по частям и атомарно сохраняет NPY-кеш.

    Args:
        model: Локальная модель SentenceTransformer.
        frame: Запросы или объявления в стабильном порядке.
        variant: Проверяемый состав текста.
        query_side: Признак кодирования запросов.
        output: Путь итоговой NPY-матрицы.

    Returns:
        Отображённую в память матрицу и диагностику длины входных текстов.

    Raises:
        OSError: Если кеш нельзя создать или опубликовать.
    """
    if cache_is_valid(output, len(frame)):
        metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
        print(f"  кеш: {output}", flush=True)
        return np.load(output, mmap_mode="r"), metadata["diagnostics"]

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".part.npy")
    matrix = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(len(frame), VECTOR_SIZE)
    )
    started = time.monotonic()
    word_counts: list[int] = []
    empty = 0
    for start in range(0, len(frame), ENCODE_CHUNK_SIZE):
        stop = min(start + ENCODE_CHUNK_SIZE, len(frame))
        texts = compose_texts(frame.iloc[start:stop], variant, query_side)
        counts = [len(text.split()) for text in texts]
        word_counts.extend(counts)
        empty += sum(count == 0 for count in counts)
        matrix[start:stop] = model.encode(
            texts,
            batch_size=ENCODE_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)
        matrix.flush()
        print(f"  {output.stem}: {stop:,}/{len(frame):,}", flush=True)
    del matrix
    os.replace(temporary, output)
    diagnostics = {
        "mean_whitespace_tokens": float(np.mean(word_counts)),
        "empty_texts": int(empty),
        "seconds": time.monotonic() - started,
    }
    metadata = {
        "rows": len(frame),
        "columns": VECTOR_SIZE,
        "model_revision": MODEL_REVISION,
        "diagnostics": diagnostics,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return np.load(output, mmap_mode="r"), diagnostics
