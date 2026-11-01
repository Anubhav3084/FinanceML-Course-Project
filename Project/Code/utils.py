import numpy as np
import pandas as pd
from datetime import datetime, timedelta

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
    pd_window_end_date = (pd_window_start_date + pd.DateOffset(months=7)).date()

    if pd.notna(def_date) and def_date <= pd_window_start_date: return pd.NA
    return int(pd.notna(def_date) and (pd_window_start_date <= def_date <= pd_window_end_date))

if __name__ == '__main__':
    pass