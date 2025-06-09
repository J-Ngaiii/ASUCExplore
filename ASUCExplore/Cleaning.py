import numpy as np
import pandas as pd
import re

_valid_iterables = (list, tuple, pd.Series, np.ndarray, pd.Index) # dictionary key and value objects are NOT valid iterables because they cannot be indexed into

def get_valid_iter():
    return _valid_iterables

def _is_type(inpt, t):
    # private function
    """
    Private helper function to check if an input is of a specified type or, if iterable, 
    whether all elements belong to at least one specified type.

    Args:
        inpt: The input value or iterable of values to be checked.
        t: A single type or an iterable of types to validate against.

    Returns:
        bool: True if `inpt` matches at least one of the specified types, 
              or if `inpt` is an iterable and all its elements match at least one type in `t`. 
              False otherwise.
    """
    def _is_type_helper(inpt, t):
        """
        Checks if the input is of a specified type `t` or, if an iterable, 
        whether all elements in `inpt` are of type `t`.
        """
        if isinstance(inpt, get_valid_iter()) and len(inpt) == 0:
            raise ValueError("Input iterable to check types of is an empty iterable.")
        return isinstance(inpt, t) or (isinstance(inpt, get_valid_iter()) and all(isinstance(x, t) for x in inpt))
    
    if isinstance(t, get_valid_iter()):
        if len(t) == 0:
            raise ValueError("Type input 't' is an empty iterable.")
        return any(_is_type_helper(inpt, type) for type in t) #was previously all
    else:
        return _is_type_helper(inpt, t)
    
def is_type(inpt, t):
    """
    Public function to check if an input is of a specified type or, if iterable, 
    whether all elements belong to at least one specified type.

    Args:
        inpt: The input value or iterable of values to be checked.
        t: A single type or an iterable containing multiple types to validate against.

    Returns:
        bool: 
            - True if `inpt` is of type `t`.
            - True if `inpt` is an iterable and all its elements match at least one type in `t`.
            - False otherwise.

    Examples:
        >>> is_type(5, int)
        True
        
        >>> is_type([1, 2, 3], int)
        True
        
        >>> is_type(["hello", 3], (int, str))
        True
        
        >>> is_type(["hello", 3], int)
        False
    """
    
    return _is_type(inpt, t)
    
def _in_df(inpt, df):
    #private function
    """
    Private function to check if a given input (column label or index) exists in a DataFrame.

    Args:
        inpt: A string (column label), an integer (column index), or an iterable (tuple, list, or pd.Series) of strings or integers.
        df (pd.DataFrame): The DataFrame to check against.

    Returns:
        bool: 
            - True if `inpt` is a column label in `df` (if `inpt` is a string).
            - True if `inpt` is a valid column index in `df` (if `inpt` is an integer and non-negative).
            - True if all elements in `inpt` exist as column labels in `df` (if `inpt` is an iterable of strings).
            - True if all elements in `inpt` are valid column indices in `df` (if `inpt` is an iterable of non-negative integers).
            - False otherwise.

    Raises:
        AssertionError: If `inpt` is not a string, integer, or an iterable of strings/integers.
        AssertionError: If `inpt` is a negative integer.
    """
    assert is_type(inpt, (str, int)), 'inpt must be string, int or tuple, list or pd.Series of strings or ints.'
    if isinstance(inpt, str): 
        return inpt in df.columns
    elif isinstance(inpt, int):
        assert inpt >= 0, 'integer inpt values must be non-negative.'
        return inpt < len(df.columns)
    elif isinstance(inpt[0], str):
        return pd.Series(inpt).isin(df.columns).all()
    elif isinstance(inpt[0], int):
        return all(pd.Series(inpt) < len(df.columns))


def in_df(inpt, df):
    """
    Public function to check if a given input (column label or index) exists in a DataFrame.

    This function wraps `_in_df()`.

    Args:
        inpt: A string (column label), an integer (column index), or an iterable (tuple, list, or pd.Series) of strings or integers.
        df (pd.DataFrame): The DataFrame to check against.

    Returns:
        bool: True if `inpt` exists in the DataFrame as a column label or index, False otherwise.

    Examples:
        >>> in_df("ColumnA", df)
        True
        
        >>> in_df([0, 2, 4], df)
        True
    """
    return _in_df(inpt, df)
    
def _any_in_df(inpt, df):
    #private function
    """
    Private function to check if at least one column in an iterable exists in a DataFrame.

    This function does not handle integers because DataFrame shape can be used to check if an index exists.

    Args:
        inpt: A string (column label) or an iterable (tuple, list, or pd.Series) of strings.
        df (pd.DataFrame): The DataFrame to check against.

    Returns:
        bool: 
            - True if `inpt` is a column label in `df` (if `inpt` is a string).
            - True if at least one element in `inpt` exists as a column label in `df` (if `inpt` is an iterable of strings).
            - False otherwise.

    Raises:
        AssertionError: If `inpt` is not a string or an iterable of strings.
    """
    assert is_type(inpt, str), 'inpt must be string or tuple, list or pd.Series of strings.'
    if isinstance(inpt, str): 
        return inpt in df.columns
    else:
        return any(df.columns.isin(inpt))
    
def any_in_df(inpt, df):
    """
    Public function to check if at least one column in an iterable exists in a DataFrame.

    This function wraps `_any_in_df()`.

    Args:
        inpt: A string (column label) or an iterable (tuple, list, or pd.Series) of strings.
        df (pd.DataFrame): The DataFrame to check against.

    Returns:
        bool: True if at least one element in `inpt` exists in the DataFrame as a column label, False otherwise.

    Examples:
        >>> any_in_df("ColumnA", df)
        True
        
        >>> any_in_df(["ColumnA", "NonexistentColumn"], df)
        True
    """
    return _any_in_df(inpt, df)
    
def _concatonater(input_df, base_df, sort_cols=None):
    #private
    output = pd.concat([input_df, base_df])
    if sort_cols is not None:
        assert is_type(sort_cols, str), 'sort_cols must be string or a list of strings'
        assert in_df(list(sort_cols), base_df) or in_df(list(sort_cols), input_df), 'Column/some columns in sort_cols not in input_df or base_df.'
        
        output = output.sort_values(by=sort_cols, ascending=False)
    
    return output

def concatonater(input_df, base_df, sort_cols=None):
    return _concatonater(input_df, base_df, sort_cols)

def _academic_year_parser(inpt):
    #private
    def _academic_year_helper(timestamp):
        """Takes timestamp and returns academic year"""
        if timestamp.month > 7:
            return str(timestamp.year) + "-" + str(timestamp.year + 1)
        elif timestamp.month < 6:
            return str(timestamp.year - 1) + "-" + str(timestamp.year)
        else:
            raise ValueError("The input timestamp occurs in June or July, which are typically not part of a standard academic year.")
        
    def _validate_and_parse(data):
        """Validates and parses collections of timestamps."""
        try:
            timestamps = pd.Series(data).apply(pd.Timestamp)
        except Exception:
            raise ValueError("At least one input could not be converted to a valid timestamp.")
        if not timestamps.apply(lambda x: hasattr(x, "month") and hasattr(x, "year")).all():
            raise ValueError("At least one timestamp does not include both month and year attributes.")
        return timestamps.apply(_academic_year_helper)
        
    if isinstance(inpt, pd.Timestamp):
        return _academic_year_helper(inpt)
    elif isinstance(inpt, str):
        try:
            inpt = pd.Timestamp(inpt)
        except Exception as e:
            raise ValueError('inpt could not be converted to valid timestamp')
        if hasattr(inpt, "month") and hasattr(inpt, "year"):
            return _academic_year_helper(inpt)
        else:
            raise ValueError("The timestamp must include both month and year attributes.")
    elif isinstance(inpt, get_valid_iter()):
        return _validate_and_parse(inpt)
    else:
        raise ValueError("Input must be a string, pd.Timestamp, or a list, tuple, or pd.Series containing strings or pd.Timestamps.")
    
def academic_year_parser(inpt):
    """Takes in a date and returns which academic year it's a part of."""
    return _academic_year_parser(inpt)

def _reverse_academic_year_parser(inpt, year_start_end):
    assert is_type(inpt, str), 'Input must be a string or list of strings specifying academic year'

    def _acayear_instance_processor(inpt, year_start_end):
        if '-' in inpt:
            years = inpt.split('-')
            start = pd.Timestamp(f"{years[0]}-{year_start_end[0][0]}-{year_start_end[0][1]}")
            end = pd.Timestamp(f"{years[1]}-{year_start_end[1][0]}-{year_start_end[1][1]}")
        elif 'fy' in inpt.lower():
            end_year = f"20{inpt[2:]}"
            start_year = str(int(end_year) - 1)
            start = pd.Timestamp(f"{start_year}-{year_start_end[0][0]}-{year_start_end[0][1]}")
            end = pd.Timestamp(f"{end_year}-{year_start_end[1][0]}-{year_start_end[1][1]}")
        else:
            raise ValueError('Inputted academic year belongs to unkown format.')
        return (start, end)
    if isinstance(inpt, str):
        return _acayear_instance_processor(inpt, year_start_end)
    elif isinstance(inpt, get_valid_iter()):
        return list(_reverse_academic_year_parser(np.array(inpt)))
    else: 
        raise ValueError("Input must be a string, or a list os strings indicating academic year in a known format.")

def reverse_academic_year_parser(inpt, year_start_end=((8, 15), (5, 20))):
    """Takes in a academic year and returns a tuple indicating date range that pertains to said academic year.
    If there are multiple academic years, returns as a list of 2d tuples.

    year_start_end: requires you to specify the month and date of the start of the academic year and end of 
    the academic year in a nested tuple format ((start_month, start_day), (end_month, end_day))
    
    Handles following formats:
    - year-year (eg. 2023-2024)
    - FY## (eg. FY24 corrsponding to 2023-2024)"""
    return _reverse_academic_year_parser(inpt, year_start_end)

def type_test(df, str_cols=None, int_cols=None, float_cols=None, date_cols=None):
    # public function
    """
    Function checks the types of all entries in designated columns for the inputted dataframe.
    Checks if designated columns contain only the designated datatype or NaN values.
    """

    if (str_cols is None) and (int_cols is None) and (float_cols is None) and (date_cols is None):
        raise ValueError('No columns to check inputted')    
    if str_cols is not None:
        if df[str_cols].applymap(lambda x: isinstance(x, str) or pd.isna(x)).all().all(): print(f"String test complete with no errors!")
        else: print(f"ERROR string columns do not have just strings or NaN values")
    if int_cols is not None: 
        if df[int_cols].applymap(lambda x: isinstance(x, int) or pd.isna(x)).all().all(): print(f"Int test complete with no errors!")
        else: print(f"ERROR int columns do not have just int or NaN values")
    if float_cols is not None:
        if df[float_cols].applymap(lambda x: isinstance(x, float) or pd.isna(x)).all().all(): print(f"Float test complete with no errors!")
        else: print(f"ERROR float columns do not have just float or NaN values")
    if date_cols is not None:
        if df[date_cols].applymap(lambda x: isinstance(x, pd.Timestamp) or pd.isna(x)).all().all(): print(f"Datetime test complete with no errors!")
        else: print(f"ERROR datetime columns do not have just datetime or NaN values")

def row_test(cleaned_df, raw_df=None, num=None):
    # public function
    """
    Checks to see if the #rows in the original dataframs is the same as the #rows in the cleaned dataframe.
    """
    
    if (raw_df is None) and (num is None):
        raise ValueError('No 2nd or 3rd argument detected. Please input either raw dataframe or specified number of rows to check cleaned dataframe against.')
    if (raw_df is not None) and (num is not None):
        raise ValueError('Row test cannot take three inputs. Either compare against raw dataframe or input number of rows cleaned dataframe is suppoed to have')
    
    if raw_df is not None: 
        if raw_df.shape[0] == cleaned_df.shape[0]: print(f"Number of rows consistent at {cleaned_df.shape[0]}")
        else: print(f"ERROR number of rows inconsistent, raw df has {raw_df.shape[0]} rows while cleaned df has {cleaned_df.shape[0]} rows")
    if num is not None:
        if num == cleaned_df.shape[0]: print(f"Number of rows consistent, df has {cleaned_df.shape[0]} rows")
        else: print(f"ERROR number of columns inconsistent, supposed to be {num} but df actually has {cleaned_df.shape[0]} rows") 

def col_test(cleaned_df, raw_df=None, num=None):
    # public function
    """
    Checks to see if the columns in the original dataframs is the same as the columns in the cleaned dataframe and/or if they have the same number of columns
    """
    
    if (raw_df is None) and (num is None):
        raise ValueError('No 2nd or 3rd argument detected. Please input either raw dataframe or specified number of columns to check cleaned dataframe against.')
    
    if raw_df is not None: 
        if (raw_df.columns == cleaned_df.columns).all(): print(f"Name of columns consistent, df has columns {list(cleaned_df.columns)}")
        else: print(f"ERROR name of columns inconsistent, raw df has columns {list(raw_df.columns)} while cleaned df has columns {list(cleaned_df.columns)}")
    if num is not None:
        if num == len(cleaned_df.columns): print(f"Number of columns consistent, df has {len(cleaned_df.columns)} columns")
        else: print(f"ERROR number of columns inconsistent, supposed to be {num} but df actually has {len(cleaned_df.columns)} columns") 

def col_mismatch_test(df1, df2, print_matches=False, d1offset=0):
    # public function
    """
    Takes in two dataframes with semi-sorted columns and checks if a the nth column in df1 is the same as the nth column in df2.
    If not then checks if a column of the same name is in df2 at all but just in a different spot

    Note this isn't a fully comprehensive test that checks every combination of possible matches and mismatches, it's just here to make spotting matched and mismatched columns easier.
    Burden of figuring out how to resolve mismatched or extra columns is on the programmer.
    """
    
    df1_cols = pd.Series(df1.columns[d1offset:])
    df2_cols = pd.Series(df2.columns)

    # Shape comparison and initialization
    same_shape = len(df1_cols) == len(df2_cols)
    excluded = None
    if len(df1_cols) > len(df2_cols):
        match = df1_cols[:len(df2_cols)].equals(df2_cols)
        excluded = df1_cols[len(df2_cols):]
        larger = 'df1'
    elif len(df1_cols) < len(df2_cols):
        match = df1_cols.equals(df2_cols[:len(df1_cols)])
        excluded = df2_cols[len(df1_cols):]
        larger = 'df2'
    else:
        match = df1_cols.equals(df2_cols)
    
    # Display results
    if match:
        if same_shape:
            print("No mismatches in columns between df1 and df2")
        else:
            print(f"No mismatches in examined columns but {larger} columns {excluded} excluded due to mismatched shape")
    else:
        if not same_shape:
            print(f"Shape discrepancy: {larger} is the larger dataframe by {np.abs(len(df1_cols) - len(df2_cols))} column(s)")
        for i in range(min(len(df1_cols), len(df2_cols))):
            if df1_cols[i] == df2_cols[i]:
                if print_matches:
                    print(f"Column {i+1} MATCHES ('{df1_cols[i]}' for df1 and '{df2_cols[i]}' for df2)")
            else:
                if df1_cols[i] in df2_cols.values:
                    print(f"Column {i+1} MISMATCH ('{df1_cols[i]}' for df1 and '{df2_cols[i]}' for df2) but '{df1_cols[i]}' IS in df2")
                else:
                    print(f"Column {i+1} MISMATCH ('{df1_cols[i]}' for df1 and '{df2_cols[i]}' for df2) and '{df1_cols[i]}' NOT in df2")

    # If extra columns in the larger dataframe, list them
    if excluded is not None and len(excluded) > 0:
        print(f"Extra columns in {larger}: {list(excluded)}")

def cat_migration_checker(df1, df2, match_col, migrating_col, trans_analysis=False):
    """
    Analyzes changes in categories or statuses of clubs between two years, identifying changes, inactivity, and new entries.

    Parameters:
    df1 (pd.DataFrame): DataFrame representing the latest year.
    df2 (pd.DataFrame): DataFrame representing the previous year.
    match_col (str): Column used to match entries between the two DataFrames.
    migrating_col (str): Column indicating categories to analyze for migration.
    trans_analysis (bool): If True, analyzes and prints category transitions.

    Returns:
    tuple: A tuple containing the following DataFrames:
        - merged: Merged DataFrame containing entries from both years.
        - no_change: Entries that did not change categories.
        - migrated: Entries that changed categories.
        - died: Entries that became inactive or disappeared.
        - birthed: Entries that became active or appeared.

    Examples:
    >>> df1 = pd.DataFrame({
    ...     'Org ID': [1, 2, 3],
    ...     'Category': ['A', 'B', 'C'],
    ...     'Year': [2023, 2023, 2023],
    ...     'Active': [1, 1, 1]
    ... })
    >>> df2 = pd.DataFrame({
    ...     'Org ID': [1, 2, 4],
    ...     'Category': ['A', 'C', 'D'],
    ...     'Year': [2024, 2024, 2024],
    ...     'Active': [1, 1, 1]
    ... })
    >>> merged, no_change, migrated, died, birthed = cat_migration_checker(df1, df2, 'Org ID', 'Category')
    """
    merged = pd.merge(df1, df2, on=match_col, how='outer', suffixes=('_latest', '_prev'))
    
    no_delete = merged[~merged[migrating_col + '_latest'].isna() & ~merged[migrating_col + '_prev'].isna()]
    no_change = no_delete[no_delete[migrating_col + '_latest'] == no_delete[migrating_col + '_prev']]
    migrated = no_delete[no_delete[migrating_col + '_latest'] != no_delete[migrating_col + '_prev']]

    died_A = merged[(~merged[migrating_col + '_prev'].isna()) & (merged[migrating_col + '_latest'].isna())]
    died_B = merged[(merged['Active_prev'] == 1) & (merged['Active_latest'] == 0)]
    died = pd.concat([died_A, died_B]).drop_duplicates()

    birthed_A = merged[(merged[migrating_col + '_prev'].isna()) & (~merged[migrating_col + '_latest'].isna())]
    birthed_B = merged[(merged['Active_prev'] == 0) & (merged['Active_latest'] == 1)]
    birthed = pd.concat([birthed_A, birthed_B]).drop_duplicates()
    print(f"""
          {len(no_change)} clubs did not change categories. 
          {len(migrated)} clubs changed categories
          {len(died)} clubs 'died' (inactive or missing in df1: {merged.loc[0,'Year_latest']})
          {len(birthed)} clubs 'created' (inactive or missing in df2: {merged.loc[0,'Year_prev']}) 
          """)
    
    if trans_analysis:
        Transition = migrated[migrating_col + '_prev'] + " --> TO --> " + migrated[migrating_col + '_latest']
        print(f"""
        ==========================================================================================
        Number of transitions are as follows (left is df2 ({merged.loc[0,'Year_prev']}) category, right is df1 ({merged.loc[0,'Year_latest']}) category):
        
        {Transition.value_counts()}
        """)
    return merged, no_change, migrated, died, birthed
