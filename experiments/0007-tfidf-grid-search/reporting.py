"""Persist point metrics, bootstrap summaries, and the compact report."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import pandas as pd

from settings import (
    BOOTSTRAP_RUNS,
    BOOTSTRAP_SEED,
    CANDIDATE_POOL,
    CHAR_NGRAMS,
    CHAR_WEIGHTS,
    LOCATION_BOOSTS,
    NORMALIZATIONS,
    TOP_K,
    WORD_NGRAMS,
)


CONFIG_COLUMNS = [
    "normalization",
    "word_ngram",
    "char_ngram",
    "char_weight",
    "location_boost",
]


def _bootstrap_summary(bootstrap: pd.DataFrame) -> pd.DataFrame:
    runs = bootstrap.pivot_table(
        index=CONFIG_COLUMNS + ["bootstrap_run"],
        columns="mode",
        values="recall_at_50",
    ).reset_index()
    runs["mix"] = 0.63 * runs["cold"] + 0.37 * runs["warm"]
    pieces = []
    for value in ("cold", "warm", "mix"):
        summary = (
            runs.groupby(CONFIG_COLUMNS, sort=False)[value]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
            .rename(
                columns={
                    statistic: f"bootstrap_{value}_{statistic}"
                    for statistic in ("mean", "std", "min", "max")
                }
            )
        )
        pieces.append(summary)
    result = pieces[0]
    for piece in pieces[1:]:
        result = result.merge(piece, on=CONFIG_COLUMNS, how="inner")
    return result


def save_results(
    output_dir: Path,
    result_rows: list[dict[str, object]],
    bootstrap_rows: list[dict[str, object]],
    validations: dict[str, pd.DataFrame],
    char_diagnostics: list[dict[str, object]],
    word_diagnostics: list[dict[str, object]],
) -> pd.DataFrame:
    metrics = pd.DataFrame(result_rows)
    comparison = metrics.pivot_table(
        index=CONFIG_COLUMNS, columns="mode", values="recall_at_50"
    ).reset_index()
    comparison["benchmark_mix_63_cold_37_warm"] = (
        0.63 * comparison["cold"] + 0.37 * comparison["warm"]
    )
    comparison = comparison.merge(
        _bootstrap_summary(pd.DataFrame(bootstrap_rows)),
        on=CONFIG_COLUMNS,
        how="inner",
    ).sort_values(CONFIG_COLUMNS, kind="stable")
    comparison.to_csv(output_dir / "bootstrap_summary.csv", index=False)

    metrics = metrics.merge(
        comparison.drop(columns=["cold", "warm"]),
        on=CONFIG_COLUMNS,
        how="left",
    ).sort_values(CONFIG_COLUMNS + ["mode"], kind="stable")
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    ranked = comparison.sort_values(
        "bootstrap_mix_mean", ascending=False, kind="stable"
    ).reset_index(drop=True)
    report = _build_report(
        ranked, validations, char_diagnostics, word_diagnostics
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ranked


def _build_report(
    ranked: pd.DataFrame,
    validations: dict[str, pd.DataFrame],
    char_diagnostics: list[dict[str, object]],
    word_diagnostics: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "grid": {
            "normalizations": list(NORMALIZATIONS),
            "word_ngrams": {key: list(value) for key, value in WORD_NGRAMS.items()},
            "char_ngrams": {key: list(value) for key, value in CHAR_NGRAMS.items()},
            "char_weights": CHAR_WEIGHTS,
            "location_boosts": LOCATION_BOOSTS,
            "configurations": len(ranked),
        },
        "bootstrap": {
            "runs": BOOTSTRAP_RUNS,
            "seed": BOOTSTRAP_SEED,
            "sampling": "query-level with replacement, separately for cold and warm",
            "ranking_metric": "bootstrap_mix_mean",
        },
        "validation": {
            "cold_queries": len(validations["cold"]),
            "warm_queries": len(validations["warm"]),
            "candidate_pool": CANDIDATE_POOL,
            "top_k": TOP_K,
            "mixture": {"cold": 0.63, "warm": 0.37},
        },
        "libraries": {
            "scikit-learn": importlib.metadata.version("scikit-learn"),
            "snowballstemmer": importlib.metadata.version("snowballstemmer"),
            "pymorphy3": importlib.metadata.version("pymorphy3"),
            "pymorphy3-dicts-ru": importlib.metadata.version("pymorphy3-dicts-ru"),
        },
        "char_indexes": char_diagnostics,
        "word_indexes": word_diagnostics,
        "best_config": ranked.iloc[0].to_dict(),
        "top_configs": ranked.head(20).to_dict("records"),
    }
