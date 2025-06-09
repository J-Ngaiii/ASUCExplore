import numpy as np
import pandas as pd
import re
# import spacy
# nlp_model = spacy.load("en_core_web_md")
from sklearn.metrics.pairwise import cosine_similarity 
# from rapidfuzz import fuzz, process

from ASUCExplore.Cleaning import is_type, in_df, any_in_df, reverse_academic_year_parser, get_valid_iter

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
    assert in_df(['Active', 'Year', 'Year Rank', 'OASIS RSO Designation'], OASIS_master), "'Year', 'Year Rank', 'Active' or 'OASIS RSO Designation' columns not found in inputted OASIS dataset."
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

def sucont_cleaner(df, year):
    """Version 1.0: Just handles cleaning for years"""
    assert ('Date' in df.columns) and is_type(df['Date'], pd.Timestamp), 'df must have "Date" column that contains only pd.Timestamp objects'
    
    copy = df.copy()
    year_range = reverse_academic_year_parser(year)
    mask = (copy['Date'] >= year_range[0]) & (copy['Date'] <= year_range[1])
    return pd.DataFrame(copy[mask])

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

#we will use the updated 2024-2025 categories and modify our 2023-2024 dataset to match it
def category_updater(df1, df2):
    """
    Updates the 'OASIS RSO Designation' column in df1 based on a mapping from df2 using Org ID.
    
    The function performs the following steps:
    1. Creates a copy of the first DataFrame (`df1`).
    2. Maps values from `df2['Organization Name_latest']` to `df2['OASIS RSO Designation_latest']`.
    3. Updates the 'OASIS RSO Designation' in `df1` based on this mapping.
    4. Fills any remaining missing ('NaN') values in the 'OASIS RSO Designation' column 
       with the corresponding values from the original `df1['OASIS RSO Designation']`.
    5. Returns the updated DataFrame and the indices of rows where the 'OASIS RSO Designation' was successfully updated.

    Args:
        df1 (pd.DataFrame): The original DataFrame to update, containing the 'Organization Name' and 'OASIS RSO Designation' columns.
        df2 (pd.DataFrame): The DataFrame containing the updated mappings from 'Organization Name_latest' to 'OASIS RSO Designation_latest'.
        
    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The updated DataFrame with the 'OASIS RSO Designation' column updated.
            - pd.Index: The indices of the rows where the 'OASIS RSO Designation' column was successfully updated (i.e., the values were not NaN).
    """
    cop = df1.copy()
    update_map = dict(zip(df2['Org ID'], df2['OASIS RSO Designation_latest']))
    cop['OASIS RSO Designation'] = cop['Org ID'].map(update_map).fillna(df1['OASIS RSO Designation'])
    
    indices = cop[cop['OASIS RSO Designation'] != df1['OASIS RSO Designation']].index #indices of all non-NaN values (the values we changed)
    print(f'{len(indices)} values updated')
    return cop, indices

def heading_finder(df, start_col, start, nth_start = 0, shift = 0, start_logic = 'exact', 
                   end_col = None, end = None, nth_end = 0, end_logic = 'exact') -> pd.DataFrame:
    """
    Non-destructively adjusts the DataFrame to start at the correct header. Can also specify where to end the new outputted dataframe.
    Last two arguments 'start_logic' and 'end_logic' allow for 'exact' or 'contains' matching logics for the header value specified in 'start' and the ending value in 'end' we're looking for.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - start_col (str or int): Column index or name to search for the header.
    - start (str): The name of the header/string to look for in the 'start_col' column where we want to start the new dataframe.
    - nth_start (int): If there are multiple occurences of 'start' in 'start_col', begin our new dataframe at the 'nth_start' occurences of 'start' in 'col'.
    - shift (int, optional): Dictates how many rows below the header row the new dataframe should start at. 
        Default is 0 which means that the extracted start value becomes the header (ie the row corresponding to the 'nth_start match in 'start_col' is set as the header).
    
    - end_col (str or int): Column index or name to search for the ending value.
    - end (str, int, or list, optional): The ending value(s) or row index to limit the DataFrame.
    - nth_end (int): If there are multiple occurences of 'end' in 'col' start at the 'nth_start' occurences of 'header' in 'col'.

    - start_logic (str, optional): Matching method for the `start` value. Default is exact matching.
        start logic options implemented: 'exact', 'contains', 'in'.
     - end_logic (str, optional): Matching method for the `end` value.  Default is exact matching.
        ending logic options implemented: 'exact', 'contains', 'in'.

    Returns:
    - pd.DataFrame: The adjusted DataFrame starting from the located header and ending at the specified end.
    """
    
    def _get_loc_wrapper(df, index_iter, elem=None):
        """returns numerical index or list of numerical indices corresponding to a non-numerical index or list of non-numerical indices
        index_iter: the instance or list of non-numerical indices to be converted into numerical integer indices"""
        
        if isinstance(index_iter, get_valid_iter()):
            assert all(index_iter.isin(df.index)), f"Not all entries in 'index_iter' under 'heading_finder' > '_get_loc_wrapper' are found in inputted df.index : {df.index}"
        
        if isinstance(index_iter, int):
            return df.index.get_loc(index_iter)
        else: 
            index_iter = pd.Series(index_iter)
            try:
                print(f"get col index itter: {index_iter}")
                indices = pd.Series(index_iter).apply(lambda x: df.index.get_loc(x)).sort_values()
                if indices.empty:
                    raise ValueError('_get_loc_wrapper indices negative')
                if elem is None: 
                    return indices
                else: 
                    assert is_type(elem, int), "Inputted 'elem' arg must be int or list of int specifyig indices to extract."
                    return indices[elem]
            except Exception as e:
                raise e

    assert isinstance(start_col, str) or isinstance(start_col, int), "'start_col' must be index of column or name of column."
    assert in_df(start_col, df), 'Given start_col is not in the given df.'

    if end_col is not None:
        assert isinstance(end_col, str) or isinstance(end_col, int), "'end_col' must be index of column or name of column."
        assert in_df(end_col, df), 'Given end_col is not in the given df.'
    else:
        end_col = start_col

    start_col_index = df.columns.get_loc(start_col) if isinstance(start_col, str) else start_col #extract index of start and end column
    end_col_index = df.columns.get_loc(end_col) if isinstance(end_col, str) else end_col #extract index of start and end column


    # print(f"start_col_index value: {start_col_index}")
    # print(f"Label: {start}")
    # print(f"Inputted index iter: {df[df.iloc[:, start_col_index].str.strip() == start].index}")
    
    if start_logic == 'exact':
        matching_indices: pd.Index = df[df.iloc[:, start_col_index].astype(str).str.strip() == str(start)].index 
    elif start_logic == 'contains':
        matching_indices: pd.Index = df[df.iloc[:, start_col_index].astype(str).str.strip().str.contains(str(start), regex=False, na=False)].index 
    else: 
        raise ValueError("Invalid 'start_logic'. Use 'exact' or 'contains'.") 

    if matching_indices.empty:
        raise ValueError(f"Header '{start}' not found in column '{start_col}'.")

    start_index: int = df.index.get_loc(matching_indices[nth_start]) #select nth_start if multiple matches with header exist

    # print(f"type start: {type(start_index)}")
    # print(f"type shift: {type(shift)}")
    start_index = start_index + shift
    if start_index >= len(df):
        raise ValueError("Shifted start index exceeds DataFrame length.")

    df = df.iloc[start_index:]

    if end is not None:
        if isinstance(end, int):
            if end < len(df):
                return df.iloc[:end]
            raise ValueError("Ending index exceeds the remaining DataFrame length.")

        elif isinstance(end, get_valid_iter()): # if 'end' is a iterable containing values to end by we want to iterate through it
            pattern = '|'.join(map(str, end))
            if end_logic == 'exact':
                end_matches = df[df.iloc[:, end_col_index].isin(end)].index
            elif end_logic == 'contains':
                end_matches = df[df.iloc[:, end_col_index].fillna('').str.contains(pattern, na=False)].index
            else:
                raise ValueError("Invalid 'end_logic'. Use 'exact' or 'contains'.")
        elif isinstance(end, str):
            if end_logic == 'exact':
                end_matches = df[df.iloc[:, end_col_index] == str(end)].index
            elif end_logic == 'contains':
                end_matches = df[df.iloc[:, end_col_index].fillna('').str.contains(str(end), na=False)].index
            else:
                raise ValueError("Invalid 'end_logic'. Use 'exact' or 'contains'.")
        else: # brute force try to convert dataframe and 'end' input into a string to match
            if end_logic == 'exact':
                end_matches = df[df.iloc[:, end_col_index].astype(str) == str(end)].index
            elif end_logic == 'contains':
                end_matches = df[df.iloc[:, end_col_index].fillna('').astype(str).str.contains(str(end), na=False)].index
            else:
                raise ValueError("Invalid 'end_logic'. Use 'exact' or 'contains'.")

        if not end_matches.empty:
            end_index = df.index.get_loc(end_matches[nth_end])
            rv = df.iloc[:end_index]
            rv_header = df.iloc[0,:]
            rv = rv[1:]
            rv.columns = rv_header
            return rv
        raise ValueError(f"End value '{end}' not found in column '{end_col}'.")

    rv = df
    rv_header = df.iloc[0,:]
    rv = rv[1:]
    rv.columns = rv_header
    return df
