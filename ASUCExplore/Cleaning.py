from collections.abc import Iterable
import pandas as pd

def is_valid_iter(inpt, exclude = None):
    """
    Checks if a certain data type is a 'valid iterable' meaning that it belongs to the Iterable class and can be indexed.
    """
    if exclude is None:
        return isinstance(inpt, Iterable) and hasattr(inpt, "__getitem__")
    else:
        if isinstance(exclude, Iterable):
            return all(isinstance(inpt, Iterable) and hasattr(inpt, "__getitem__") and not isinstance(inpt, ex) for ex in exclude)
        else:
            return isinstance(inpt, Iterable) and hasattr(inpt, "__getitem__") and not isinstance(inpt, exclude)

def _is_type(inpt, t, report):
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
    def _is_type_helper(inpt, t, report):
        """
        Checks if the input is of a specified type `t` or, if an iterable, 
        whether all elements in `inpt` are of type `t`.
        """
        if isinstance(inpt, Iterable) and len(inpt) == 0:
            if report:
                print(f"WARNING: Input is an empty iterable '{inpt}' but asked to check for type {t}.")
            return False
        return isinstance(inpt, t) or (isinstance(inpt, Iterable) and all(isinstance(x, t) for x in inpt)) # handles case where inpt is a string --> we return isinstance(inpt, t) before iterating through it
    
    if isinstance(t, Iterable):
        if len(t) == 0:
            raise ValueError(f"Iterable {t} passed in for types to check for but iterable was empty.")
        return any(_is_type_helper(inpt, type, report) for type in t) #was previously all
    else:
        return _is_type_helper(inpt, t, report)
    
def is_type(inpt, t, report=False):
    """
    Public function to check if an input is of a specified type or, if iterable, 
    whether all elements belong to at least one specified type.

    Args:
        inpt: The input value or iterable of values to be checked.
        t: A single type or an iterable containing multiple types to validate against.
            - if an iterable of types is passed through the function checks if the input is the same type/an iterable with all elements the same type as at least one of the types listed in 't'

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
    assert not isinstance(t, (str, bytes)), "'t' arg cannot be a string or bytes or else we iterate through individual characters"
    if isinstance(t, Iterable):
        t = tuple(t)

    try:
        if isinstance(inpt, t): # Direct type check: handles inpt and t both are not iterables
            return True
        elif any(isinstance(inpt, elem) for elem in t):
            return True # Handles inpt not iterable, t iterable
    except TypeError:  # Catch only type-related errors
        return _is_type(inpt, t, report=report)
    
    if isinstance(inpt, Iterable): # Handles inpt iterable, t not iterable as well as both inpt and t iterable
        return _is_type(inpt, t, report=report)
    
def in_df(inpt, df):
    """
    Function to check if a given input (column label or index) exists in a DataFrame. Though handling indices is kinda dumb just check the number of columns fr fr. 

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
    elif isinstance(inpt, Iterable):
        if isinstance(inpt[0], str):
            return pd.Series(inpt).isin(df.columns).all()
        elif isinstance(inpt[0], int):
            return all(pd.Series(inpt) < len(df.columns))
    
def any_in_df(inpt, df):
    """
    Function to check if at least one column in an iterable exists in a DataFrame.

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
    assert is_type(inpt, str), 'inpt must be string or iterable of strings.'
    if isinstance(inpt, str): 
        return inpt in df.columns
    elif isinstance(inpt, Iterable):
        inpt = list(inpt)
        return any(df.columns.isin(inpt))
    
def any_drop(df, cols):
    """
    Drops any and all columns instantiated in the 'cols' arg from 'df' arg if they're present."""
    assert isinstance(df, pd.DataFrame), f"Inputted 'df' should be a pandas dataframe,  but is {type(df)}"
    if list(cols) != []:
        assert is_type(cols, str), "'cols' must be a string or an iterable of strings."
        assert any_in_df(cols, df), f"None of the columns in {cols} are present in the DataFrame."
    else:
        return df

    if isinstance(cols, str):
        cols_to_drop = [cols] if cols in df.columns else []
    else:
        cols_to_drop = [c for c in cols if c in df.columns]
    return df.drop(columns=cols_to_drop)

