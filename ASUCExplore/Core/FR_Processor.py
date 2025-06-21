import pandas as pd
from ASUCExplore.Utils import heading_finder
from ASUCExplore.Cleaning import in_df

def FR_Helper(df, given_start = 'Appx', start_col = 0, adding_end_keyword='END', end_col = 0, alphabet=None, nth_occurence = 1, reporting=False) -> pd.DataFrame:
    """
    Returns rows of dataframe that correspond to an given alphabet. Stops at first NA row if no alphabet is provided.
    """
    assert isinstance(start_col, str) or isinstance(start_col, int), "'start_col' must be index of column or name of column."
    assert in_df(start_col, df), f"start_col '{start_col}' is not in df columns: {df.columns.tolist()}"

    def in_alphabet_helper(series: pd.Series, alphabet_list: list[str], nth: int = 1) -> pd.Series:
        occurrences = dict.fromkeys(alphabet_list, 0)
        result_mask = []
        for val in series:
            if val in occurrences:
                if occurrences[val] < nth:
                    result_mask.append(True)
                    occurrences[val] += 1
                else:
                    result_mask.append(False)
            else:
                result_mask.append(False)
        return pd.Series(result_mask, index=series.index)

    if end_col is not None:
        assert isinstance(end_col, str) or isinstance(end_col, int), "'end_col' must be index of column or name of column."
        assert in_df(end_col, df), 'Given end_col is not in the given df.'
    else:
        end_col = start_col

    start_col_index = df.columns.get_loc(start_col) if isinstance(start_col, str) else start_col #extract index of start and end column
    end_col_index = df.columns.get_loc(end_col) if isinstance(end_col, str) else end_col #extract index of start and end column
    
    copy = df.copy()
    copy = heading_finder(copy, start_col=0, start=given_start, start_logic='contains') # no ending logic just take all rows below the starting point
    col = copy.columns[start_col_index]
    try:
        if alphabet is None:
            na_indices = copy[copy[col].isna()].index
            if na_indices.empty:
                raise ValueError("No NaN row found to mark as end of section.")
            ending_row_index = na_indices[0]
        else:
            mask = in_alphabet_helper(copy[col], alphabet.copy(), nth_occurence)
            valid_rows: pd.DataFrame = copy[mask]
            if valid_rows.empty:
                raise ValueError("No valid rows with alphabet keys found.")
            ending_row_index = valid_rows.index[-1] + 1

    except Exception as e:
           print(f"Warning: Could not insert ending keyword '{adding_end_keyword}' in column {end_col}, received exception\n{e}")
    rv = copy[copy.index < ending_row_index]
    return rv

def FR_ProcessorV2(df, debug=False):
    """Employs heading_finder to clean data."""
    assert isinstance(df, pd.DataFrame), f'Inputted df is not a dataframe but type {type(df)}'

    FY24_Alphabet = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'.split()
    FY24_Alphabet.extend(
        'AA AB AC AD AE AF AG AH AI AJ AK AL AM AN AO AP AQ AR AS AT AU AV AW AX AY AZ'.split()
    )
    FY24_Alphabet.extend(
        'BB CC DD EE FF GG HH II JJ KK LL MM NN OO PP QQ RR SS TT UU VV WW XX YY ZZ'.split()
    )
    try:
        rv = FR_Helper(df, alphabet=FY24_Alphabet)
    except Exception as e:
        if debug:
            print(f"FR_ProcessorV2 errored on df\n{df}")
        raise e
    return rv
    