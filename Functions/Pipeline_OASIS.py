import numpy as np
import pandas as pd

from .Cleaning import in_df

def _year_adder(df_list, year_list, year_rank):
        #private
        """
        Takes a list of dataframes and a corresponding list of years, 
        then mutates those dataframes with a year column containing the year in a element-wise fashion
        """

        for i in range(len(df_list)):
            df_list[i]['Year'] = np.full(df_list[i].shape[0], year_list[i])
            df_list[i]['Year Rank'] = np.full(df_list[i].shape[0], year_rank[i])

def year_adder(df_list, year_list, year_rank):
    return _year_adder(df_list, year_list, year_rank)

def year_rank_collision_handler(df, existing):
    """For re-adjusting year rank via comparing academic year columns that have values formatted "2023-2024".
    Just remaps the Year and Year Rank columns, can handle extra columns."""
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


