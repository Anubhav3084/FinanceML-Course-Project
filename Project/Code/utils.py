import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def create_target_variable(stmt_date: str, def_date :str):
    """
        stmt_date: statement_date,
        def_date: default_date

        returns: 1/0
        Since the maximum lag can be of 7 months we consider default in (stmt_date + 7 months, stmt_date + 19 months)
        Example:
            - stmt_date = 20xx-12-31  xx = [07, 08, 09, 10]
            - def_date = 15 Oct 2010
        Output:
            - stmt_date             PD window             Target
            - 2007-12-31      2008-08-01 - 2009-07-31            0
            - 2008-12-31      2009-08-01 - 2010-07-31            0
            - 2009-12-31      2010-08-01 - 2011-07-31            1
            - 2010-12-31      2011-08-01 - 2012-07-31     drop from training
    """
    if isinstance(stmt_date, str): stmt_date = datetime.strptime(stmt_date, '%Y-%m-%d').date()
    if isinstance(def_date, str): def_date = datetime.strptime(def_date, '%Y-%m-%d').date()

    pd_window_start_date = (stmt_date + pd.DateOffset(months=7) + timedelta(days=1)).date()
    pd_window_end_date = (pd_window_start_date + pd.DateOffset(months=12)).date()

    if pd.notna(def_date) and def_date <= pd_window_start_date: return pd.NA
    return int(pd.notna(def_date) and (pd_window_start_date <= def_date <= pd_window_end_date))

def precision_recall_by_bin(df, feature, target="default", n_bins=10, higher_is_riskier=False):
    data = df[[feature, target]].copy()#.dropna(subset=[target]).copy()

    # create bins and aggregate them
    bins = pd.qcut(data[feature], q=n_bins, duplicates="drop")
    data["bin"] = bins

    result = (
        data.groupby("bin", observed=True)[target]
        .agg(records="size", defaults="sum")
        .reset_index()
        .sort_values("bin", ascending=not higher_is_riskier)
    )

    # create a bin for missing value
    result["bin"] = result["bin"].astype(object)
    missing = data[feature].isna()

    if missing.any():
        result = pd.concat([
            result,
            pd.DataFrame([{
                "bin": "Missing",
                "records": missing.sum(),
                "defaults": data.loc[missing, target].sum()
            }])
        ], ignore_index=True)

    result = result.reset_index(drop=True)
    total_defaults = data[target].sum()
    overall_rate = data[target].mean()

    # calculate precision recall
    result["population (%)"] = result["records"] / len(data)
    result["precision (%)"] = result["defaults"] / result["records"]
    result["recall (%)"] = result["defaults"] / total_defaults if total_defaults else np.nan

    # calculate cumulative precision recall
    cumulative_records = result["records"].cumsum()
    cumulative_defaults = result["defaults"].cumsum()

    result["cumm_precision (%)"] = cumulative_defaults / cumulative_records
    result["cumm_recall (%)"] = (
        cumulative_defaults / total_defaults if total_defaults else np.nan
    )

    # calculate lift and cumulative lift
    result["lift"] = np.round(result["cumm_precision (%)"] / overall_rate, 3) if overall_rate else np.nan

    result["population (%)"] = result["population (%)"].apply(lambda x: f"{x:.2%}")

    result["precision (%)"] = result["precision (%)"].apply(lambda x: f"{x:.2%}")
    result["recall (%)"] = result["recall (%)"].apply(lambda x: f"{x:.2%}")

    result["cumm_precision (%)"] = result["cumm_precision (%)"].apply(lambda x: f"{x:.2%}")
    result["cumm_recall (%)"] = result["cumm_recall (%)"].apply(lambda x: f"{x:.2%}")

    return result

def IV(df, feature, target='default'):
    df_feat = df[[feature, target]].copy()
    df_feat['bins'] = pd.qcut(df_feat[feature], q=10, duplicates='drop').astype(str)
    df_feat['bins'] = df_feat['bins'].where(df_feat[feature].notna(), "Missing")

    iv_bin = df_feat.groupby('bins')['default'].agg(Total='count', Bads='sum').reset_index()
    iv_bin['Goods'] = iv_bin['Total'] - iv_bin['Bads']
    iv_bin = iv_bin[['bins', 'Goods', 'Bads']]
    iv_bin['Goods (%)'] = (iv_bin['Goods'] + 0.01) / iv_bin['Goods'].sum()
    iv_bin['Bads (%)'] = (iv_bin['Bads'] + 0.01) / iv_bin['Bads'].sum()
    iv_bin['WoE'] = np.log(iv_bin['Goods (%)'] / iv_bin['Bads (%)'])
    iv_bin['IV_bin'] = (iv_bin['Goods (%)'] - iv_bin['Bads (%)']) * iv_bin['WoE']
    iv = float(iv_bin['IV_bin'].sum())
    return iv, iv_bin

def plot_woe_iv(iv_table, feature_name="Feature", figsize=(5, 5)):
    table = iv_table.copy()

    # Keep numeric intervals in ascending order; missing bins last
    intervals = table["bins"].map(lambda value: value.left if hasattr(value, "left") else np.inf)
    table = table.iloc[np.argsort(intervals.to_numpy(), kind="stable")]

    x = np.arange(len(table))
    labels = table["bins"].astype(str)
    total_iv = table["IV_bin"].sum()

    fig, ax_woe = plt.subplots(figsize=figsize)
    ax_iv = ax_woe.twinx()

    bars = ax_woe.bar(x, table["WoE"], width=0.65, color="steelblue", alpha=0.75, label="WoE")
    line, = ax_iv.plot(x, table["IV_bin"], color="darkorange", marker="o", linewidth=2, label="IV contribution")

    ax_woe.axhline(0, color="gray", linewidth=0.8)
    ax_woe.set_xlabel("Bins")
    ax_woe.set_ylabel("Weight of Evidence", color="steelblue")
    ax_iv.set_ylabel("IV contribution", color="darkorange")

    ax_woe.tick_params(axis="y", labelcolor="steelblue")
    ax_iv.tick_params(axis="y", labelcolor="darkorange")
    ax_iv.set_ylim(bottom=0)

    ax_woe.set_xticks(x)
    ax_woe.set_xticklabels(labels, rotation=90, ha="right")
    ax_woe.set_title(f"{feature_name} — Total IV: {total_iv:.4f}")

    ax_woe.legend(handles=[bars, line], labels=["WoE", "IV contribution"], loc="best")

    fig.tight_layout()
    return fig, (ax_woe, ax_iv)

if __name__ == '__main__':
    pass