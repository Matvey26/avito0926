#!/usr/bin/env python3
"""Скачивает закреплённую ревизию модели для эксперимента 0010.

Скрипт является единственным сетевым этапом. Он сохраняет полный snapshot
Hugging Face в ``assets/models``; основной эксперимент загружает его только из
локального каталога и поэтому воспроизводится без внешнего API.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download

from settings import DEFAULT_MODEL_PATH, MODEL_ID, MODEL_REVISION, MODEL_SHA256


REVISION_MARKER = ".experiment_revision"
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "modules.json",
    "tokenizer.json",
)


def file_sha256(path: Path) -> str:
    """Вычисляет SHA-256 файла модели порциями.

    Args:
        path: Путь к проверяемому файлу.

    Returns:
        Шестнадцатеричный SHA-256 нижнего регистра.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_complete(path: Path) -> bool:
    """Проверяет наличие обязательных файлов и маркера ревизии.

    Args:
        path: Локальный каталог snapshot модели.

    Returns:
        ``True``, если каталог соответствует закреплённой ревизии и содержит
        все файлы, необходимые для offline-загрузки SentenceTransformer.
    """
    marker = path / REVISION_MARKER
    return (
        marker.exists()
        and marker.read_text(encoding="utf-8").strip() == MODEL_REVISION
        and all((path / filename).exists() for filename in REQUIRED_FILES)
        and file_sha256(path / "model.safetensors") == MODEL_SHA256
    )


def download(output: Path) -> None:
    """Скачивает полный snapshot и записывает маркер успешной подготовки.

    Args:
        output: Итоговый локальный каталог модели.

    Returns:
        Ничего.

    Raises:
        OSError: Если каталог нельзя создать или snapshot нельзя записать.
    """
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=output,
    )
    missing = [name for name in REQUIRED_FILES if not (output / name).exists()]
    if missing:
        raise RuntimeError(f"В snapshot отсутствуют файлы: {missing}")
    actual_hash = file_sha256(output / "model.safetensors")
    if actual_hash != MODEL_SHA256:
        raise RuntimeError(f"Неверный SHA-256 весов модели: {actual_hash}")
    (output / REVISION_MARKER).write_text(MODEL_REVISION + "\n", encoding="utf-8")
    print(f"Модель {MODEL_ID}@{MODEL_REVISION} сохранена в {output}")


def main() -> None:
    """Разбирает аргументы и при необходимости подготавливает модель.

    Returns:
        Ничего.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    if is_complete(args.output):
        print(f"Модель уже загружена и проверена: {args.output}")
        return
    download(args.output)


if __name__ == "__main__":
    main()
