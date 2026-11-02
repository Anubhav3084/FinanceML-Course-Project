from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm


def add_target_column(df, fully_observed_years=range(2007, 2012)):
    """Create 12-month target: start 7 months + 1 day after stmt_date,
    ending 19 months after stmt_date. Early defaults and unobserved years
    receive NA; later defaults/no recorded defaults receive 0.
    """
    d = df.copy()
    stmt = pd.to_datetime(d["stmt_date"], errors="coerce")
    default = pd.to_datetime(d["def_date"], errors="coerce")

    d["window_start"] = stmt + pd.DateOffset(months=7, days=1)
    d["window_end"] = stmt + pd.DateOffset(months=19)

    valid_window = d["window_start"].notna() & d["window_end"].notna()
    year_observed = d["fs_year"].isin(fully_observed_years) & valid_window
    d["early_default"] = default.notna() & (default < d["window_start"])
    in_window = default.between(d["window_start"], d["window_end"])

    d["target"] = pd.Series(pd.NA, index=d.index, dtype="Int64")
    d.loc[year_observed & in_window, "target"] = 1
    d.loc[year_observed & ~d["early_default"] & ~in_window, "target"] = 0

    d["prediction_window"] = (
        d["window_start"].dt.strftime("%Y-%m-%d")
        + " to "
        + d["window_end"].dt.strftime("%Y-%m-%d")
    )
    return d


def peer_impute(
    data,
    reference,
    numeric_cols,
    id_col="id",
    sector_col="ateco_sector",
    city_col="HQ_city",
    asset_col="asst_tot",
):
    """Impute numeric missing values using only reference (training) peers."""
    out = data.copy()
    ref = reference.copy()

    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    ref[numeric_cols] = ref[numeric_cols].replace([np.inf, -np.inf], np.nan)
    out[asset_col] = pd.to_numeric(out[asset_col], errors="coerce")
    ref[asset_col] = pd.to_numeric(ref[asset_col], errors="coerce")
    medians = ref[numeric_cols].median()
    cache = {}
    counts = {}

    def get_pool(feature, group_col, group_value):
        key = (feature, group_col, group_value)
        if key not in cache:
            mask = (
                ref[group_col].eq(group_value).fillna(False)
                & ref[feature].notna()
                & ref[asset_col].notna()
                & ref[asset_col].ge(0)
            )
            pool = ref.loc[mask, [id_col, asset_col, feature]].copy()
            pool = pool.drop_duplicates(id_col, keep="last")
            pool["_log_assets"] = np.log1p(pool[asset_col])
            nn = None
            if len(pool):
                nn = NearestNeighbors(metric="euclidean").fit(
                    pool[["_log_assets"]]
                )
            cache[key] = (pool, nn)
        return cache[key]

    total = int(out[numeric_cols].isna().sum().sum())
    with tqdm(total=total, desc="Peer imputation", unit="cell") as bar:
        for feature in numeric_cols:
            for idx in out.index[out[feature].isna()]:
                row = out.loc[idx]
                firm_id, asset = row[id_col], row[asset_col]
                value, method = np.nan, None

                if pd.notna(asset) and asset >= 0:
                    # First try same-sector peers, requiring at least five.
                    if pd.notna(row[sector_col]):
                        pool, nn = get_pool(feature, sector_col, row[sector_col])
                        peers = pool.loc[pool[id_col].ne(firm_id)]
                        n_peers = len(peers)
                        if n_peers >= 5:
                            k_pct = math.ceil(0.05 * n_peers)
                            k = max(5, k_pct)
                            method = "sector_5pct" if k_pct > 5 else "sector_5"
                            n_query = min(len(pool), k + 1)
                            query = pd.DataFrame({"_log_assets": [np.log1p(asset)]})
                            ix = nn.kneighbors(
                                query, n_neighbors=n_query, return_distance=False
                            )[0]
                            donors = pool.iloc[ix]
                            donors = donors.loc[donors[id_col].ne(firm_id)].head(k)
                            if len(donors):
                                value = donors[feature].median()

                    # If sector imputation did not work, try same-city peers.
                    if pd.isna(value) and pd.notna(row[city_col]):
                        pool, nn = get_pool(feature, city_col, row[city_col])
                        peers = pool.loc[pool[id_col].ne(firm_id)]
                        n_peers = len(peers)
                        if n_peers:
                            k_pct = math.ceil(0.05 * n_peers)
                            k = min(n_peers, max(5, k_pct))
                            method = "city_5pct" if k_pct > 5 else "city_peers"
                            n_query = min(len(pool), k + 1)
                            query = pd.DataFrame({"_log_assets": [np.log1p(asset)]})
                            ix = nn.kneighbors(
                                query, n_neighbors=n_query, return_distance=False
                            )[0]
                            donors = pool.iloc[ix]
                            donors = donors.loc[donors[id_col].ne(firm_id)].head(k)
                            if len(donors):
                                value = donors[feature].median()

                if pd.isna(value):
                    value = medians[feature]
                    method = "training_median"

                if pd.notna(value):
                    out.at[idx, feature] = value
                    key = (feature, method)
                    counts[key] = counts.get(key, 0) + 1
                bar.update(1)

    stats = pd.DataFrame(
        [{"feature": f, "method": m, "imputed_rows": n}
         for (f, m), n in counts.items()]
    )
    return out, stats


def preprocess_and_save(
    input_csv,
    output_dir="pd_preprocessed",
    fully_observed_years=range(2007, 2012),
):
    """Create labeled annual train/evaluation pairs, impute them, and save CSVs.

    Saves 2007->2008, 2008->2009, 2009->2010 validation, and final
    2007-2010->2011 holdout pairs in separate subfolders.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_csv)
    d = add_target_column(raw, fully_observed_years=fully_observed_years)

    # Exclude unlabeled rows, including early defaults and fs_year 2012.
    model_df = d.loc[d["target"].isin([0, 1])].copy()

    categorical_cols = ["ateco_sector", "HQ_city", "legal_struct"]
    for col in categorical_cols:
        model_df[col] = (
            model_df[col].astype("string").fillna("__MISSING__").astype(str)
        )

    excluded = {
        "Unnamed: 0", "id", "stmt_date", "def_date", "fs_year", "target",
        "window_start", "window_end", "prediction_window", "early_default",
        "days_rec", "eqty_corp_family_tot", *categorical_cols,
    }
    numeric_cols = [
        c for c in model_df.select_dtypes(include="number").columns
        if c not in excluded
    ]
    if "asst_tot" not in numeric_cols:
        raise ValueError("asst_tot is required for peer imputation.")

    folds = [
        ("2007_train_2008_test", [2007], 2008),
        ("2008_train_2009_test", [2008], 2009),
        ("2009_train_2010_validation", [2009], 2010),
        ("final_2007_2010_train_2011_test", [2007, 2008, 2009, 2010], 2011),
    ]

    summary = []
    for name, train_years, eval_year in folds:
        train = model_df.loc[model_df["fs_year"].isin(train_years)].copy()
        evaluation = model_df.loc[model_df["fs_year"].eq(eval_year)].copy()

        if train.empty or evaluation.empty:
            raise ValueError(f"{name}: training or evaluation set is empty.")
        if train["target"].nunique() < 2:
            raise ValueError(f"{name}: training set does not contain both classes.")

        # Sort so each firm’s last training-year row is its donor row.
        reference = train.sort_values(["id", "fs_year"])
        train_imp, train_stats = peer_impute(train, reference, numeric_cols)
        eval_imp, eval_stats = peer_impute(evaluation, reference, numeric_cols)

        fold_dir = output_dir / name
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_imp.to_csv(fold_dir / "train.csv", index=False)
        eval_imp.to_csv(fold_dir / "evaluation.csv", index=False)
        train_stats.to_csv(fold_dir / "train_imputation_stats.csv", index=False)
        eval_stats.to_csv(fold_dir / "evaluation_imputation_stats.csv", index=False)

        summary.append({
            "fold": name,
            "train_rows": len(train_imp),
            "train_defaults": int(train_imp["target"].sum()),
            "evaluation_year": eval_year,
            "evaluation_rows": len(eval_imp),
            "evaluation_defaults": int(eval_imp["target"].sum()),
            "train_numeric_missing_after_imputation":
                int(train_imp[numeric_cols].isna().sum().sum()),
            "evaluation_numeric_missing_after_imputation":
                int(eval_imp[numeric_cols].isna().sum().sum()),
        })

    summary = pd.DataFrame(summary)
    summary.to_csv(output_dir / "fold_summary.csv", index=False)
    return summary


if __name__ == "__main__":
    result = preprocess_and_save(
        input_csv="train-2025.csv",
        output_dir="pd_preprocessed",
    )
    print(result)

