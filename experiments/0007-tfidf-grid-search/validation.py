"""Deterministic validation construction and query-level bootstrap weights."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd

from settings import QUERY_COLUMNS, SEED
from text_processing import normalize_text


def stable_hash(value: str) -> str:
    return hashlib.sha1(f"{SEED}|{value}".encode()).hexdigest()


def signature_key(row: pd.Series) -> tuple[object, ...]:
    return tuple(row[column] for column in QUERY_COLUMNS)


def build_validation(
    train: pd.DataFrame, mode: str, size: int
) -> tuple[pd.DataFrame, dict[tuple[object, ...], set[str]]]:
    """Reproduce the fixed cold/warm proxy protocol used since experiment 0003."""
    grouped = train.groupby(QUERY_COLUMNS, sort=False, dropna=False)["item_id"].agg(
        lambda values: set(values)
    )
    signatures_by_text: dict[str, list[tuple[object, ...]]] = defaultdict(list)
    for key in grouped.index:
        signatures_by_text[normalize_text(key[0])].append(key)
    if mode == "cold":
        texts = list(signatures_by_text)
    elif mode == "warm":
        texts = [text for text, keys in signatures_by_text.items() if len(keys) >= 2]
    else:
        raise ValueError(mode)
    selected_texts = sorted(texts, key=stable_hash)[:size]
    selected_keys = [
        min(
            signatures_by_text[text],
            key=lambda key: stable_hash(repr(tuple(str(value) for value in key))),
        )
        for text in selected_texts
    ]
    validation = pd.DataFrame(selected_keys, columns=QUERY_COLUMNS)
    return validation, {key: grouped.loc[key] for key in selected_keys}


def build_bootstrap_weights(size: int, runs: int, seed: int) -> np.ndarray:
    """Return per-query multiplicities for deterministic sampling with replacement."""
    rng = np.random.default_rng(seed)
    weights = np.zeros((runs, size), dtype=np.int16)
    for run in range(runs):
        sample = rng.integers(0, size, size=size)
        weights[run] = np.bincount(sample, minlength=size)
    return weights


def target_category(value: object) -> int:
    category = int(value)
    return 114 if category == 0 else category
