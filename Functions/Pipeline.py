import numpy as np
import pandas as pd
import re
import spacy
nlp_model = spacy.load("en_core_web_md")
from sklearn.metrics.pairwise import cosine_similarity 
from rapidfuzz import fuzz, process

from .Utils import column_converter, column_renamer, oasis_cleaner
from .Cleaning import is_type, in_df, concatonater, academic_year_parser
from .Pipeline_OASIS import heading_finder, year_rank_collision_handler
from .Pipeline_Ficomm import cont_approval, close_match_sower, sa_filter




def SU_Cont_Processor(df):
    """
    Expected Intake: Df with following columns: 
    """

def OASIS_Standard_Processor(df, year, rename=None, col_types=None, existing=None):
    """
    Expected Intake: 
    - Df with following columns: ['Org ID', 'Organization Name', 'All Registration Steps Completed?',
       'Reg Form Progress\n\n (Pending means you need to wait for OASIS Staff to approve your Reg form)',
       'Number of Signatories\n(Need 4 to 8)', 'Completed T&C', 'Org Type',
       'Callink Page', 'OASIS RSO Designation', 'OASIS Center Advisor ',
       'Year', 'Year Rank']
    - existing_df: already cleaned version of OASIS dataset

    - year is to be a tuple containing the string description of the academic year and the year rank in a tuple 

    EXTRA COLUMNS ARE HANDLED BY JUST CONCATING AND LETTING NAN VALUES BE.
    """
    cleaned_df = heading_finder(df, 0, 'Org ID') #phase 1

    if rename is None: #phase 2
        cleaned_df = column_renamer(cleaned_df, 'OASIS-Standard') 
    else: 
        cleaned_df = column_renamer(cleaned_df, rename)

    cleaned_df['Year'] = year[0] #phase 3
    cleaned_df['Year Rank'] = year[1]
    
    if col_types is None: #phase 4
        OClean_Str_Cols = ['Org ID', 'Organization Name', 'Reg Steps Complete', 'Reg Form Progress', 'Completed T&C', 'Org Type',
       'Callink Page', 'OASIS RSO Designation', 'OASIS Center Advisor', 'Year']
        OClean_Int_Cols = ['Num Signatories', 'Year Rank']
        column_converter(cleaned_df, OClean_Int_Cols, int)
        column_converter(cleaned_df, OClean_Str_Cols, str)
    else:
        #expecting col_types to be a dictionary mapping type to column names
        for key in col_types.keys(): 
            column_converter(cleaned_df, col_types[key], key)
    
    cleaned_df['Active'] = cleaned_df['Org Type'].apply(lambda x: 1 if x == 'Registered Student Organizations' else 0) #phase 5

    cleaned_df['OASIS RSO Designation'] = cleaned_df['OASIS RSO Designation'].str.extract(r'[LEAD|OASIS] Center Category: (.*)') #phase 6
    
    if existing is not None:
        assert in_df(
            ['Org ID', 'Organization Name', 'Reg Steps Complete',
       'Reg Form Progress', 'Num Signatories', 'Completed T&C', 'Org Type',
       'Callink Page', 'OASIS RSO Designation', 'OASIS Center Advisor', 'Year',
       'Year Rank', 'Orientation Attendees', 'Spring Re-Reg. Eligibility',
       'Active']
            , existing) #consider functionality for telling which columns are not in df
        cleaned_df, existing = year_rank_collision_handler(cleaned_df, existing)
        cleaned_df = concatonater(cleaned_df, existing, ['Year Rank', 'Organization Name'])
        return cleaned_df
    else: 
        return cleaned_df

def Ficomm_Dataset_Processor(inpt_agenda, inpt_FR, inpt_OASIS, close_matching=True, valid_cols=None):
    """
    Expected Intake: Df with following columns: 
    """
    #phase 1: pre-processing
    inpt_agenda = cont_approval(inpt_agenda) #process agenda
    inpt_agenda['Year'] = academic_year_parser(inpt_agenda['Ficomm Meeting Date']) #add year column
    inpt_OASIS['Organization Name'] = inpt_OASIS['Organization Name'].str.strip() #strip names
    inpt_agenda['Organization Name'] = inpt_agenda['Organization Name'].str.strip()
    inpt_OASIS = oasis_cleaner(inpt_OASIS, True, list(inpt_agenda['Year'].unique()))

    #phase 2: matching and cleaning
    df = pd.merge(inpt_OASIS, inpt_agenda, on=['Organization Name', 'Year'], how='right') #initial match

    if valid_cols is None: #column cleaning
        #standard settings is to use the standard column layout
        standard_ficomm_layout = ['Org ID', 'Organization Name',	'Org Type',	'Callink Page',	'OASIS RSO Designation', 'OASIS Center Advisor', 'Year', 'Year Rank', 'Active', 'Ficomm Meeting Date', 'Ficomm Decision', 'Amount Allocated']
        assert in_df(standard_ficomm_layout), "Standard columns not detected in DF"
        df = df[standard_ficomm_layout]
    else: 
        assert is_type(valid_cols, str)
        assert in_df(valid_cols), "Inputted columns for arg 'valid_cols' not detected in DF"
        df = df[valid_cols]

    if close_matching:
        rez, fuzz_failed_match = close_match_sower(df, inpt_OASIS, 'Organization Name', 'Org Type', 87.9,  sa_filter) #optimal settings based on empirical testing
        print(f"""Failed Matches: 
              {fuzz_failed_match}""")
        
