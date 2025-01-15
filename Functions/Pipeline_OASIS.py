import numpy as np
import pandas as pd

from .Cleaning import in_df

def year_adder(df_list, year_list, year_rank):
        #private
        """
        Takes a list of dataframes and a corresponding list of years, 
        then mutates those dataframes with a year column containing the year in a element-wise fashion
        """

        for i in range(len(df_list)):
            df_list[i]['Year'] = np.full(df_list[i].shape[0], year_list[i])
            df_list[i]['Year Rank'] = np.full(df_list[i].shape[0], year_rank[i])

def heading_finder(df, col, inpt):
    """If a input's header is moved down a couple rows, checks for the where the correct header is and adjusts the dataframe to start at the right header."""
    assert isinstance(col, str) or isinstance(col, int), 'col must be index of column or name of column.'
    assert in_df(col, df), 'Given col is not in the given df.'
    if isinstance(col, str):
        col_index = df.columns.get_loc(col)
    else:
        col_index = col
    for i in range(len(df)):
        curr_entry = df.iloc[i,col_index].strip()
        if curr_entry == inpt:
            return df.iloc[i:,:]
    raise ValueError(f"Header '{inpt}' not found in column '{col}'.")

def year_rank_collision_handler(df, existing):
    """For re-adjusting year rank via comparing academic year columns that have values formatted "2023-2024"."""
    assert in_df(['Year', 'Year Rank'], df), 'Year and Year Rank not in df.'
    assert in_df(['Year', 'Year Rank'], existing), 'Year and Year Rank not in existing.'
    df_cop = df.copy()
    existing_cop = existing.copy()
    
    all_academic_years = pd.concat([existing_cop['Year'], df_cop['Year']]).unique()
    in_order = sorted(all_academic_years, key=lambda x: int(x.split('-')[1]))

    years_to_rank = {year: rank for rank, year in enumerate(in_order)}

    df_cop['Year Rank'] = df_cop['Year'].map(years_to_rank)
    existing_cop['Year Rank'] = existing_cop['Year'].map(years_to_rank)

    return df_cop, existing_cop
