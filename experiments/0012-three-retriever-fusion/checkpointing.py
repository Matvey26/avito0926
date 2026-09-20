"""Сохраняет и загружает top-500 кандидатов retrieval-каналов.

Крупные NumPy-массивы находятся в ``assets``. Метаданные содержат fingerprint
корпуса и валидационных запросов, поэтому кеш от другого разбиения не будет
молча использован в текущем эксперименте.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from settings import CANDIDATE_POOL, QUERY_COLUMNS


@dataclass(frozen=True)
class CandidateBatch:
    """Хранит кандидатов и исходные score одного сплита.

    Attributes:
        indices: Номера строк корпуса с padding ``-1``.
        scores: Исходные score retrieval в том же порядке.
        lengths: Фактическое число кандидатов каждого запроса.
    """

    indices: np.ndarray
    scores: np.ndarray
    lengths: np.ndarray

    def row(self, row_number: int) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает непустую часть одной строки кандидатов.

        Args:
            row_number: Номер запроса внутри сплита.

        Returns:
            Индексы кандидатов и их исходные score без padding.
        """
        length = int(self.lengths[row_number])
        return self.indices[row_number, :length], self.scores[row_number, :length]


def empty_batch(query_count: int) -> CandidateBatch:
    """Создаёт пустое прямоугольное хранилище top-500.

    Args:
        query_count: Число запросов в сплите.

    Returns:
        Batch с индексами ``-1``, нулевыми score и длинами.
    """
    return CandidateBatch(
        indices=np.full((query_count, CANDIDATE_POOL), -1, dtype=np.int32),
        scores=np.zeros((query_count, CANDIDATE_POOL), dtype=np.float32),
        lengths=np.zeros(query_count, dtype=np.int16),
    )


def data_fingerprint(
    items: pd.DataFrame, validations: dict[str, pd.DataFrame]
) -> str:
    """Вычисляет fingerprint порядка корпуса и запросов.

    Args:
        items: Дедуплицированный корпус в рабочем порядке.
        validations: Cold/warm-запросы в рабочем порядке.

    Returns:
        SHA-256 item ID и полных сигнатур запросов.
    """
    digest = hashlib.sha256()
    item_hashes = pd.util.hash_pandas_object(items["item_id"], index=False)
    digest.update(item_hashes.to_numpy().tobytes())
    for mode in sorted(validations):
        digest.update(mode.encode())
        hashes = pd.util.hash_pandas_object(
            validations[mode][QUERY_COLUMNS], index=False
        )
        digest.update(hashes.to_numpy().tobytes())
    return digest.hexdigest()


class CandidateStore:
    """Управляет атомарными checkpoint-файлами retrieval-каналов."""

    def __init__(self, root: Path, fingerprint: str) -> None:
        """Создаёт хранилище для конкретного набора данных.

        Args:
            root: Каталог тяжёлых артефактов эксперимента.
            fingerprint: Fingerprint корпуса и валидационных запросов.
        """
        self.root = root
        self.fingerprint = fingerprint
        root.mkdir(parents=True, exist_ok=True)

    def _data_path(self, retriever: str, mode: str) -> Path:
        """Возвращает путь NPZ одного retrieval-сплита.

        Args:
            retriever: Стабильное имя retrieval-канала.
            mode: Название валидационного режима.

        Returns:
            Путь внутри каталога checkpoint.
        """
        return self.root / f"{retriever}_{mode}.npz"

    def _metadata_path(self, retriever: str) -> Path:
        """Возвращает путь JSON-метаданных канала.

        Args:
            retriever: Стабильное имя retrieval-канала.

        Returns:
            Путь JSON-файла.
        """
        return self.root / f"{retriever}.json"

    def complete(self, retriever: str, modes: tuple[str, ...]) -> bool:
        """Проверяет полноту и совместимость checkpoint канала.

        Args:
            retriever: Имя retrieval-канала.
            modes: Обязательные режимы.

        Returns:
            ``True``, если fingerprint совпадает и все массивы существуют.
        """
        metadata_path = self._metadata_path(retriever)
        if not metadata_path.exists():
            return False
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return (
            metadata.get("fingerprint") == self.fingerprint
            and metadata.get("candidate_pool") == CANDIDATE_POOL
            and all(self._data_path(retriever, mode).exists() for mode in modes)
        )

    def save(
        self,
        retriever: str,
        batches: dict[str, CandidateBatch],
        diagnostics: dict[str, object],
    ) -> None:
        """Атомарно сохраняет массивы и затем публикует метаданные.

        Args:
            retriever: Имя retrieval-канала.
            batches: Кандидаты по режимам.
            diagnostics: Параметры и длительности построения.

        Returns:
            Ничего.
        """
        for mode, batch in batches.items():
            target = self._data_path(retriever, mode)
            temporary = target.with_suffix(".part")
            with temporary.open("wb") as output:
                np.savez(
                    output,
                    indices=batch.indices,
                    scores=batch.scores,
                    lengths=batch.lengths,
                )
            os.replace(temporary, target)
        metadata = {
            "fingerprint": self.fingerprint,
            "candidate_pool": CANDIDATE_POOL,
            "diagnostics": diagnostics,
        }
        target = self._metadata_path(retriever)
        temporary = target.with_suffix(".part")
        temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, target)

    def load(
        self, retriever: str, modes: tuple[str, ...]
    ) -> dict[str, CandidateBatch]:
        """Загружает завершённые кандидаты одного канала.

        Args:
            retriever: Имя retrieval-канала.
            modes: Загружаемые режимы.

        Returns:
            Кандидаты по режимам.

        Raises:
            RuntimeError: Если checkpoint отсутствует или несовместим.
        """
        if not self.complete(retriever, modes):
            raise RuntimeError(f"Checkpoint {retriever!r} не готов")
        result = {}
        for mode in modes:
            with np.load(self._data_path(retriever, mode)) as data:
                result[mode] = CandidateBatch(
                    indices=data["indices"],
                    scores=data["scores"],
                    lengths=data["lengths"],
                )
        return result

    def diagnostics(self, retriever: str) -> dict[str, object]:
        """Читает диагностику завершённого канала.

        Args:
            retriever: Имя retrieval-канала.

        Returns:
            Словарь из JSON-метаданных.
        """
        metadata = json.loads(
            self._metadata_path(retriever).read_text(encoding="utf-8")
        )
        return metadata["diagnostics"]
