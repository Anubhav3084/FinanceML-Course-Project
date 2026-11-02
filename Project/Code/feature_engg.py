"""Build the 31 candidates in FinanceML_Features_With_References.csv.

Returns feature columns only, preserving input order and index. No fitting,
imputation, clipping, encoding or target construction occurs here. Proxy names
are intentional: validate accounting definitions before using them in a model.
Historical features use only an explicitly supplied prior-year history table.
"""

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    'log_assets', 'net_profit_assets', 'operating_profit_assets',
    'working_capital_assets', 'sales_assets', 'liabilities_assets_proxy',
    'negative_equity', 'cash_assets', 'operating_cashflow_assets',
    'operating_margin', 'interest_coverage_proxy', 'sector_category',
    'equity_assets', 'short_term_debt_assets', 'cash_short_term_debt',
    'current_ratio_proxy', 'receivables_assets', 'receivables_days_proxy',
    'cashflow_liabilities_proxy', 'cashflow_profit_gap_assets',
    'operating_loss', 'nonpositive_revenue', 'zero_financing_expense',
    'legal_structure_category', 'missing_debt_breakdown',
    'two_year_net_losses', 'scaled_profit_change',
    'change_operating_profit_assets', 'change_cash_assets',
    'revenue_growth', 'history_unavailable',
]

NUMERIC_INPUTS = [
    'asst_tot', 'profit', 'prof_operations', 'wc_net', 'rev_operating',
    'eqty_tot', 'cash_and_equiv', 'cf_operations', 'exp_financing',
    'debt_st', 'asst_current', 'AR', 'debt_bank_st', 'debt_bank_lt',
    'debt_fin_st', 'debt_fin_lt', 'AP_st', 'AP_lt',
]


def _numeric(frame, name):
    """Reject malformed strings; interpret nonfinite numeric values as missing."""
    return pd.to_numeric(frame[name], errors='raise').astype(float).replace(
        [np.inf, -np.inf], np.nan
    )


def _ratio(numerator, denominator):
    """Calculate only for finite inputs and a strictly positive denominator."""
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        result = numerator / denominator.where(denominator > 0)
    return result.replace([np.inf, -np.inf], np.nan)


def _flag(values, condition):
    """Return 0/1 with NaN when the source value is unavailable."""
    return condition.astype(float).where(values.notna())


def build_credit_features(df, *, history=None, reporting_lag_months=7):
    """Return the 31 register features as a new DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Current financial records. Required raw columns must exist; individual
        cells may be missing. id/fs_year/stmt_date are required only with history.
    history : pandas.DataFrame or None
        Explicitly permitted historical statements, NOT an evaluation batch.
        For development, history=train_raw is suitable. Only exactly fs_year-1
        is matched, and its assumed availability must precede the current
        assessment. Historical keys (id, fs_year) must be unique. Statement
        dates must be valid and agree with fs_year. No future values are filled.
        Without history, F26-F30 are NaN and history_unavailable is 1.
    reporting_lag_months : int
        Assumed lag from statement reference date to availability. Applied to
        both historical and current statements; actual release dates would be
        preferable if available. Does not address retrospective restatements.

    Notes
    -----
    F06/F13 are complementary; F03 is approximately supplied roa / 100.
    F11/F16/F19 depend on accounting definitions that need verification.
    A denominator/state issue remains NaN, never infinity or an invented zero.
    Categories remain strings for a later training-fitted encoder.
    def_date and the default label are never accessed.
    """
    required = NUMERIC_INPUTS + ['ateco_sector', 'legal_struct']
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f'Missing required raw columns: {missing}')
    if not isinstance(reporting_lag_months, int) or reporting_lag_months < 0:
        raise ValueError('reporting_lag_months must be a nonnegative integer.')

    # Work positionally so duplicate input index labels remain safe.
    raw = df.reset_index(drop=True)
    x = pd.DataFrame({c: _numeric(raw, c) for c in NUMERIC_INPUTS})
    f = pd.DataFrame(index=raw.index)
    assets = x.asst_tot
    liabilities = assets - x.eqty_tot

    f['log_assets'] = np.log(assets.where(assets > 0))
    f['net_profit_assets'] = _ratio(x.profit, assets)
    f['operating_profit_assets'] = _ratio(x.prof_operations, assets)
    f['working_capital_assets'] = _ratio(x.wc_net, assets)
    f['sales_assets'] = _ratio(x.rev_operating, assets)
    f['liabilities_assets_proxy'] = _ratio(liabilities, assets)
    f['negative_equity'] = _flag(x.eqty_tot, x.eqty_tot < 0)
    f['cash_assets'] = _ratio(x.cash_and_equiv, assets)
    f['operating_cashflow_assets'] = _ratio(x.cf_operations, assets)
    f['operating_margin'] = _ratio(x.prof_operations, x.rev_operating)
    f['interest_coverage_proxy'] = _ratio(x.prof_operations, x.exp_financing)
    f['sector_category'] = raw.ateco_sector.astype('string')
    f['equity_assets'] = _ratio(x.eqty_tot, assets)
    f['short_term_debt_assets'] = _ratio(x.debt_st, assets)
    f['cash_short_term_debt'] = _ratio(x.cash_and_equiv, x.debt_st)
    f['current_ratio_proxy'] = _ratio(x.asst_current, x.debt_st)
    f['receivables_assets'] = _ratio(x.AR, assets)
    f['receivables_days_proxy'] = 365 * _ratio(
        x.AR.where(x.AR >= 0), x.rev_operating
    )
    f['cashflow_liabilities_proxy'] = _ratio(x.cf_operations, liabilities)
    f['cashflow_profit_gap_assets'] = _ratio(x.profit - x.cf_operations, assets)
    f['operating_loss'] = _flag(x.prof_operations, x.prof_operations < 0)
    f['nonpositive_revenue'] = _flag(x.rev_operating, x.rev_operating <= 0)
    f['zero_financing_expense'] = _flag(x.exp_financing, x.exp_financing == 0)
    f['legal_structure_category'] = raw.legal_struct.astype('string')
    debt_columns = [
        'debt_bank_st', 'debt_bank_lt', 'debt_fin_st', 'debt_fin_lt',
        'AP_st', 'AP_lt',
    ]
    f['missing_debt_breakdown'] = x[debt_columns].isna().any(axis=1).astype(float)

    for name in FEATURE_NAMES[25:30]:
        f[name] = np.nan
    f['history_unavailable'] = 1.0

    if history is not None:
        h_required = [
            'id', 'fs_year', 'stmt_date', 'profit', 'prof_operations',
            'asst_tot', 'cash_and_equiv', 'rev_operating',
        ]
        for name, frame, cols in [
            ('df', raw, ['id', 'fs_year', 'stmt_date']),
            ('history', history, h_required),
        ]:
            absent = sorted(set(cols) - set(frame.columns))
            if absent:
                raise ValueError(f'{name} missing history keys/inputs: {absent}')

        h = history[h_required].copy().reset_index(drop=True)

        def keys(frame):
            years = pd.to_numeric(frame.fs_year, errors='raise')
            dates = pd.to_datetime(frame.stmt_date, errors='raise')
            if (frame.id.isna().any() or years.isna().any()
                    or dates.isna().any() or (years % 1 != 0).any()
                    or (dates.dt.year != years).any()):
                raise ValueError('History matching needs valid id, year and statement date.')
            return frame.id, years.astype('int64'), dates

        ids, years, dates = keys(raw)
        h_ids, h_years, h_dates = keys(h)
        h_index = pd.MultiIndex.from_arrays([h_ids, h_years])
        if h_index.has_duplicates:
            raise ValueError('history has duplicate (id, fs_year) keys.')
        h = h.assign(_date=h_dates).set_axis(h_index)
        lookup = pd.MultiIndex.from_arrays([ids, years - 1])
        prior = h.reindex(lookup).reset_index(drop=True)
        lag = pd.DateOffset(months=reporting_lag_months)
        eligible = prior._date.notna() & ((prior._date + lag) < (dates + lag))
        prior.loc[~eligible, ['profit', 'prof_operations', 'asst_tot',
                              'cash_and_equiv', 'rev_operating']] = np.nan
        p = pd.DataFrame({c: _numeric(prior, c) for c in [
            'profit', 'prof_operations', 'asst_tot', 'cash_and_equiv', 'rev_operating'
        ]})
        both_profits = x.profit.notna() & p.profit.notna()
        f['two_year_net_losses'] = (
            (x.profit < 0) & (p.profit < 0)
        ).astype(float).where(both_profits)
        f['scaled_profit_change'] = _ratio(
            x.profit - p.profit, x.profit.abs() + p.profit.abs()
        )
        f['change_operating_profit_assets'] = (
            f.operating_profit_assets - _ratio(p.prof_operations, p.asst_tot)
        )
        f['change_cash_assets'] = f.cash_assets - _ratio(p.cash_and_equiv, p.asst_tot)
        f['revenue_growth'] = _ratio(x.rev_operating, p.rev_operating) - 1
        f['history_unavailable'] = (~eligible).astype(float)

    # Protect against numerical overflow even in otherwise valid ratios.
    numeric = f.select_dtypes(include='number').columns
    f[numeric] = f[numeric].replace([np.inf, -np.inf], np.nan)
    f = f[FEATURE_NAMES]
    f.index = df.index.copy()
    return f
