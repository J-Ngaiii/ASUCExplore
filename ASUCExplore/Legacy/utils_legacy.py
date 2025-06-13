

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