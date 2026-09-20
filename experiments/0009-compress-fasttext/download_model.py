#!/usr/bin/env python3
"""Скачивает и проверяет русскую модель compress-fastText для 0009.

Скрипт нужен только для подготовки локального артефакта. Основной эксперимент
не обращается к сети и загружает модель из ``assets/models``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import urllib.request

from settings import DEFAULT_MODEL_PATH, MODEL_SHA256, MODEL_URL


def sha256(path: Path) -> str:
    """Вычисляет SHA-256 файла порциями.

    Аргументы:
        path: Путь к проверяемому файлу.

    Возвращает:
        Шестнадцатеричный SHA-256 нижнего регистра.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(output: Path) -> None:
    """Скачивает модель во временный файл и атомарно публикует её.

    Аргументы:
        output: Итоговый локальный путь модели.

    Возвращает:
        Ничего.

    Исключения:
        RuntimeError: Если контрольная сумма скачанного файла неверна.
        OSError: Если файл нельзя скачать или записать.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    urllib.request.urlretrieve(MODEL_URL, temporary)
    actual = sha256(temporary)
    if actual != MODEL_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Неверный SHA-256 модели: ожидался {MODEL_SHA256}, получен {actual}"
        )
    temporary.replace(output)
    print(f"Модель сохранена в {output}")


def main() -> None:
    """Разбирает аргументы и скачивает модель, если она ещё не подготовлена.

    Возвращает:
        Ничего.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    if args.output.exists() and sha256(args.output) == MODEL_SHA256:
        print(f"Модель уже загружена и проверена: {args.output}")
        return
    download(args.output)


if __name__ == "__main__":
    main()
