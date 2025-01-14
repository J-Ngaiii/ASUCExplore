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

from .Utils import column_converter, column_renamer, year_adder, heading_finder
from .Cleaning import is_type, in_df, concatonater, academic_year_parser




def SU_Cont_Processor(df):
    """
    Expected Intake: Df with following columns: 
    """

def OASIS_Standard_Processor(df, rename=None, existing=None):
    """
    Expected Intake: 
    - Df with following columns: ['Org ID', 'Organization Name', 'All Registration Steps Completed?',
       'Reg Form Progress\n\n (Pending means you need to wait for OASIS Staff to approve your Reg form)',
       'Number of Signatories\n(Need 4 to 8)', 'Completed T&C', 'Org Type',
       'Callink Page', 'OASIS RSO Designation', 'OASIS Center Advisor ',
       'Year', 'Year Rank']
    - existing_df: already cleaned version of OASIS dataset
    """
    headed_df = heading_finder(df, 0, 'Org ID')

    if rename is None:
        renamed_df = column_renamer(headed_df, 'OASIS-Standard')
    else: 
        renamed_df = column_renamer(headed_df, rename)
    
    if existing is not None:
        assert in_df(
            ['Org ID', 'Organization Name', 'Reg Steps Complete',
       'Reg Form Progress', 'Num Signatories', 'Completed T&C', 'Org Type',
       'Callink Page', 'OASIS RSO Designation', 'OASIS Center Advisor', 'Year',
       'Year Rank', 'Orientation Attendees', 'Spring Re-Reg. Eligibility',
       'Active']
            , existing)
        output = concatonater(output, existing, ['Year Rank', 'Organization Name'])
        return output
    else: 
        return output

def Ficomm_Dataset_Processor(ficomm_agenda, FR_sheets, OASISMaster):
    """
    Expected Intake: Df with following columns: 
    """