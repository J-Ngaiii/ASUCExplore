import numpy as np
import pandas as pd
import re
import spacy
nlp_model = spacy.load("en_core_web_md")
from sklearn.metrics.pairwise import cosine_similarity 
from rapidfuzz import fuzz, process

from .Cleaning import is_type, in_df, any_in_df

def column_converter(df, cols, t, datetime_element_looping = False):
    """
    Mutates the inputted dataframe 'df' but with columns 'cols' converted into type 't'.
    Can handle conversion to int, float, pd.Timestamp and str
    
    Version 1.0: CANNOT Convert multple columns to different types
    """
    

    if isinstance(cols, str): # If a single column is provided, convert to list for consistency
        cols = [cols]
    
    if t == int:
        df[cols] = df[cols].fillna(-1).astype(t)
        
    elif t == float:
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        
    elif t == pd.Timestamp:
        if not datetime_element_looping:
            for col in cols: 
                df[col] = pd.to_datetime(df[col], errors='coerce')
        else: #to handle different tries being formatted differently
            for col in cols: 
                for index in df[col].index:
                    df.loc[index, col] = pd.to_datetime(df.loc[index, col], errors='coerce')
        
    elif t == str:
        df[cols] = df[cols].astype(str)
        
    else:
        try:
            df[cols] = df[cols].astype(t)
        except Exception as e:
            print(f"Error converting {cols} to {t}: {e}")

def column_renamer(df, rename):
        """Column Renaming Unit is for renaiming columns or a df, with custom modes according to certain raw ASUC datasets files expected.
        Can handle extra columns. they just don't get renamed if they aren't explicitly named in the 'renamed' arg."""
        cleaned_df = df.copy()
        cols = cleaned_df.columns

        if rename == 'OASIS-Standard': #proceed with standard renaming scheme
            cleaned_df = cleaned_df.rename(columns={
                cols[2] : 'Reg Steps Complete',
                cols[3] : 'Reg Form Progress',
                cols[4] : 'Num Signatories',
                cols[9] : 'OASIS Center Advisor'
            }
            )
        else:
            #rename should be a dictionary of indexes to the renamed column
            assert isinstance(rename, dict), 'rename must be a dictionary mapping the index of columns/names of columns to rename to their new names'
            assert in_df(list(rename.keys()), df), 'names or indices of columns to rename must be in given df'
            if is_type(list(rename.keys()), int):
                cleaned_df = cleaned_df.rename(columns={ cols[key] : rename[key] for key in rename.keys()})
            elif is_type(list(rename.keys()), str):
                cleaned_df = cleaned_df.rename(columns={ key : rename[key] for key in rename.keys()})
        
        cleaned_df.columns = cleaned_df.columns.str.strip() #removing spaces from column names
        assert is_type(cleaned_df.columns, str), 'CRU Final Check: columns not all strings'

        return cleaned_df

def any_drop(df, cols):
    assert is_type(cols, str), "'cols' must be a string or an iterable (list, tuple, or pd.Series) of strings."
    assert any_in_df(cols, df), f"None of the columns in {cols} are present in the DataFrame."
    if isinstance(cols, str):
        cols_to_drop = [cols] if cols in df.columns else []
    else:
        cols_to_drop = df.columns[df.columns.isin(cols)]
    return df.drop(columns=cols_to_drop)


def oasis_cleaner(OASIS_master, approved_orgs_only=True, year=None, club_type=None):
    """
    Cleans the OASIS master dataset by applying filters and removing unnecessary columns.

    Version 2.0: Updated with cleaning functions from this package

    Parameters:
    OASIS_master (DataFrame): The master OASIS dataset to be cleaned.
    approved_orgs_only (bool): If True, only includes active organizations (where 'Active' == 1).
    year (int, float, str, or list, optional): The year(s) to filter by. Can be:
        - A single academic year as a string (e.g., '2023-2024').
        - A single year rank as an integer or float (e.g., 2023 or 2023.0).
        - A list of academic years or year ranks.
    club_type (str, optional): Filters by a specific club type from the 'OASIS RSO Designation' column.

    Returns:
    DataFrame: The cleaned OASIS dataset with the specified filters applied and unnecessary columns removed.

    Notes:
    - When filtering by year:
        * Strings are matched against the 'Year' column (e.g., '2023-2024').
        * Integers or floats are matched against the 'Year Rank' column (e.g., 2023).
    - The following columns are always dropped: 'Orientation Attendees', 'Spring Re-Reg. Eligibility', 
        'Completed T&C', 'Num Signatories', 'Reg Form Progress', and 'Reg Steps Complete'.

    Raises:
    TypeError: If the year is not a string, integer, float, or list of these types.
    AssertionError: If a float in the year list is not an integer (e.g., 2023.5).
    """
    if year is not None:
        assert is_type(year, (str, int, float)), "Year must be a string, integer, float, or a tuple, list or pd.Series of these types."
        if isinstance(year, (str, int, float)):
            year = [year]
        if is_type(year, float):
            for y in year:
                if y != round(y):
                    raise AssertionError("All floats in `year` must represent integers.")

    OASISCleaned = OASIS_master.copy()
    assert in_df(['Active', 'Year', 'Year Rank', 'OASIS RSO Designation']), "'Year', 'Year Rank', 'Active' or 'OASIS RSO Designation' columns not found in inputted OASIS dataset."
    if approved_orgs_only:
        OASISCleaned = OASISCleaned[OASISCleaned['Active'] == 1]

    if year is not None:
        if is_type(year, str): #at this point year should be an iterable
            OASISCleaned = OASISCleaned[OASISCleaned['Year'].isin(year)]
        elif is_type(year, int) or is_type(year, float):
            OASISCleaned = OASISCleaned[OASISCleaned['Year Rank'].isin(year)]

    if club_type is not None:
            OASISCleaned = OASISCleaned[OASISCleaned['OASIS RSO Designation'] == club_type]
    
    standard_drop_cols = ['Orientation Attendees', 'Spring Re-Reg. Eligibility', 'Completed T&C', 'Num Signatories', 'Reg Form Progress', 'Reg Steps Complete']
    if any_in_df(standard_drop_cols, OASISCleaned):
        OASISCleaned = any_drop(OASISCleaned, standard_drop_cols)
    return OASISCleaned

def bulk_manual_populater(df, override_cols, indices, override_values): 
    """
    Manually overrides specific column values in a DataFrame at specified indices with given override values.

    Parameters:
    df (pd.DataFrame): The input DataFrame to modify.
    override_cols (list): List of column names to override values in.
    indices (list): List of row indices corresponding to the override columns.
    override_values (list): List of values to override with.

    Returns:
    pd.DataFrame: A copy of the input DataFrame with the specified overrides applied.

    Raises:
    ValueError: If the lengths of `override_cols`, `indices`, and `override_values` are not the same.

    Examples:
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> override_cols = ['A', 'B']
    >>> indices = [0, 1]
    >>> override_values = [10, 20]
    >>> bulk_manual_populater(df, override_cols, indices, override_values)
       A   B
    0  10   4
    1   2  20
    2   3   6 
    """

    copy = df.copy()
    for i in range(len(override_cols)):
        copy.loc[indices[i], override_cols[i]] = override_values[i]
    return copy

