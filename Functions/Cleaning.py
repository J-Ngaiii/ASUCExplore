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

def type_test(df, str_cols=None, int_cols=None, float_cols=None, date_cols=None):
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

