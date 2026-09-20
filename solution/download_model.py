#!/usr/bin/env python3
"""Скачивает и проверяет зафиксированную модель compress-fastText GeoWAC."""

from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request

from fasttext_channel import file_sha256
from settings import DEFAULT_MODEL_PATH, FASTTEXT_MODEL_SHA256, FASTTEXT_MODEL_URL


def main() -> None:
    """Скачивает модель во временный файл и атомарно устанавливает её.

    Raises:
        RuntimeError: Если SHA-256 загруженного файла неверен.
        OSError: Если загрузка или запись файла завершилась ошибкой.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    try:
        urllib.request.urlretrieve(FASTTEXT_MODEL_URL, temporary)
        actual_hash = file_sha256(temporary)
        if actual_hash != FASTTEXT_MODEL_SHA256:
            raise RuntimeError(f"Неверный SHA-256 модели: {actual_hash}")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Модель сохранена: {args.output}")
