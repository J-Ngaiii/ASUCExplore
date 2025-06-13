import pandas as pd

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