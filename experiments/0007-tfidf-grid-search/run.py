#!/usr/bin/env python3
"""Grid-search word/char TF-IDF, morphology, blending, and location boost."""

from __future__ import annotations

import pandas as pd

from indexing import build_char_retrievals, evaluate_word_indexes
from reporting import save_results
from retrieval import build_fallbacks
from settings import (
    BOOTSTRAP_RUNS,
    BOOTSTRAP_SEED,
    ITEM_COLUMNS,
    QUERY_COLUMNS,
    parse_args,
)
from validation import build_bootstrap_weights, build_validation


def main() -> None:
    args = parse_args(__doc__)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(
        args.data_dir / "train.parquet", columns=QUERY_COLUMNS + ITEM_COLUMNS
    )
    items = train[ITEM_COLUMNS].drop_duplicates("item_id").reset_index(drop=True)
    item_categories = items["item_category_id"].to_numpy()
    item_locations = items["item_location_id"].to_numpy()
    item_row_by_id = dict(zip(items["item_id"], items.index))
    category_fallbacks, location_fallbacks = build_fallbacks(items)

    validations: dict[str, pd.DataFrame] = {}
    relevance_rows: dict[
        str, dict[tuple[object, ...], frozenset[int]]
    ] = {}
    bootstrap_weights = {}
    for mode_number, mode in enumerate(("cold", "warm")):
        validation, relevance_ids = build_validation(train, mode, args.validation_size)
        validations[mode] = validation
        relevance_rows[mode] = {
            key: frozenset(item_row_by_id[item_id] for item_id in item_ids)
            for key, item_ids in relevance_ids.items()
        }
        bootstrap_weights[mode] = build_bootstrap_weights(
            len(validation), BOOTSTRAP_RUNS, BOOTSTRAP_SEED + mode_number
        )

    char_retrievals, char_diagnostics = build_char_retrievals(
        items, validations, item_categories
    )
    result_rows, bootstrap_rows, word_diagnostics = evaluate_word_indexes(
        items,
        validations,
        relevance_rows,
        bootstrap_weights,
        char_retrievals,
        item_categories,
        item_locations,
        category_fallbacks,
        location_fallbacks,
    )
    ranked = save_results(
        args.output_dir,
        result_rows,
        bootstrap_rows,
        validations,
        char_diagnostics,
        word_diagnostics,
    )
    print("\nTop configurations by bootstrap mean:", flush=True)
    print(ranked.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
