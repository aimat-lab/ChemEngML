import numpy as np
import pandas as pd
import ast


def apply_filter(df: pd.DataFrame, string: str) -> pd.DataFrame:
    """Applies a filter on a dataframe column and returns the modified dataframe.
    The string input is parsed by a simple string split into column name, operator and value.
    Hence, when passing lists as value (e.g. "col in [1,2,3]") make sure that there are no blanks.

    Args:
        df (pd.DataFrame): Dataframe to be filtered.
        string (str): Filter as string in the format "<column name> <operator> <value>"

    Example:
        >>> df = pd.DataFrame()
        >>> df['ints'] = np.array([-2, -1, 0, 1, 2], dtype=int)
        >>> df['floats'] = np.array([-2.5, -1.5, -0.5, 0.5, 1.5], dtype=float)
        >>> df['strings'] = np.array(['a', 'b', 'c', 'd', 'e'])
        >>> filters = ["ints > -2",
                       "floats < 0.5",
                       "strings in ['a','b','c']"]
        >>> for f in filters:
                df = apply_filter(df, f)
        >>> print(df)
           ints  floats strings
        1    -1    -1.5       b
        2     0    -0.5       c
    """
    ops = {'<': lambda x, y: x < y,
           '<=': lambda x, y: x <= y,
           '>': lambda x, y: x > y,
           '>=': lambda x, y: x >= y,
           '==': lambda x, y: x == y,
           'in': lambda x, y: x.isin(y)}
    col, op, val = string.split()
    try:
        val = ast.literal_eval(val)
    except ValueError:
        pass
    return df[ops[op](df[col], val)]