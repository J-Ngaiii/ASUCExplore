import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import re
import unittest
import pytest
import spacy
nlp_model = spacy.load("en_core_web_md")
from sklearn.metrics.pairwise import cosine_similarity 
from rapidfuzz import fuzz, process

from .Cleaning import is_type, in_df

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
        """Column Renaming Unit is for renaiming columns or a df, with custom modes according to certain raw ASUC datasets files expected."""
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


