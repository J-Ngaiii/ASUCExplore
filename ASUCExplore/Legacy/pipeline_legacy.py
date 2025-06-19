import numpy as np
import pandas as pd
import re
import spacy
nlp_model = spacy.load("en_core_web_md")
from sklearn.metrics.pairwise import cosine_similarity 
from rapidfuzz import fuzz, process

from ASUCExplore.Utils import column_converter, column_renamer, oasis_cleaner, heading_finder
from ASUCExplore.Cleaning import is_type, in_df
from ASUCExplore.Core.OASIS_Processor import year_rank_collision_handler, asuc_processor
from ASUCExplore.Core.Name_Matching import close_match_sower, sa_filter
from ASUCExplore.Legacy import cont_approval, concatonater, academic_year_parser
from ASUCExplore.Core.FR_Processor import FR_Processor

def OASIS_Standard_Processor(df, year, rename=None, col_types=None, existing=None):
    """
    Expected Intake: 
    - Df with following columns: ['Org ID', 'Organization Name', 'All Registration Steps Completed?',
       'Reg Form Progress\n\n (Pending means you need to wait for OASIS Staff to approve your Reg form)',
       'Number of Signatories\n(Need 4 to 8)', 'Completed T&C', 'Org Type',
       'Callink Page', 'OASIS RSO Designation', 'OASIS Center Advisor ',
       'Year', 'Year Rank']
    - existing_df: already cleaned version of OASIS dataset
    - col_types: a dictionary mapping data types to column names, thus assigning certain/validating columns to have certain types

    - year is to be a tuple containing the string description of the academic year and the year rank in a tuple 

    EXTRA COLUMNS ARE HANDLED BY JUST CONCATING AND LETTING NAN VALUES BE.
    """
    cleaned_df = heading_finder(df, 0, 'Org ID') #phase 1

    if rename is None: #phase 2
        cleaned_df = column_renamer(cleaned_df, 'OASIS-Standard') 
    else: 
        cleaned_df = column_renamer(cleaned_df, rename)

    cleaned_df['Year'] = year[0] #phase 3: there is no info on the df that allows us to parse academic year
    cleaned_df['Year Rank'] = year[1]
    
    if col_types is None: #phase 4
        OClean_Str_Cols = ['Org ID', 'Organization Name', 'Reg Steps Complete', 'Reg Form Progress', 'Completed T&C', 'Org Type',
       'Callink Page', 'OASIS RSO Designation', 'OASIS Center Advisor', 'Year']
        OClean_Int_Cols = ['Num Signatories', 'Year Rank']
        column_converter(cleaned_df, OClean_Int_Cols, int)
        column_converter(cleaned_df, OClean_Str_Cols, str)
    else:
        #expecting col_types to be 
        for key in col_types.keys(): 
            column_converter(cleaned_df, col_types[key], key)
    
    cleaned_df['Active'] = cleaned_df['Org Type'].apply(lambda x: 1 if x == 'Registered Student Organizations' else 0) #phase 5

    cleaned_df['OASIS RSO Designation'] = cleaned_df['OASIS RSO Designation'].str.extract(r'[LEAD|OASIS] Center Category: (.*)') #phase 6
    
    if existing is not None: #phase 7(O): concating onto an existing OASIS dataset
        assert in_df(
            ['Org ID', 'Organization Name', 'Reg Steps Complete',
       'Reg Form Progress', 'Num Signatories', 'Completed T&C', 'Org Type',
       'Callink Page', 'OASIS RSO Designation', 'OASIS Center Advisor', 'Year',
       'Year Rank', 'Orientation Attendees', 'Spring Re-Reg. Eligibility',
       'Active']
            , existing), "Columns expected to be in cleaned 'existing' df non-existent."
        cleaned_df, existing = year_rank_collision_handler(cleaned_df, existing)
        cleaned_df = concatonater(cleaned_df, existing, ['Year Rank', 'Organization Name'])
        return cleaned_df
    else: 
        return cleaned_df

def Ficomm_Dataset_Processor(inpt_agenda, inpt_FR, inpt_OASIS, close_matching=True, custom_close_match_settings=None, valid_cols=None, breaking=None):
    """
    Expected Intake: Df with following columns: 
    inpt_OASIS: a master OASIS doc, it autocleans for the right year in phase 1

     - custom_close_match_settings: iterable that unpacks into arg values for close_match_sower
        - Args to fill: matching_col, mismatch_col, fuzz_threshold, filter, nlp_processing, nlp_process_threshold, nlp_threshold
    """
    #phase 1: pre-processing
    inpt_agenda = cont_approval(inpt_agenda) #process agenda
    inpt_agenda['Year'] = academic_year_parser(inpt_agenda['Ficomm Meeting Date']) #add year column
    inpt_OASIS['Organization Name'] = inpt_OASIS['Organization Name'].str.strip() #strip names
    inpt_agenda['Organization Name'] = inpt_agenda['Organization Name'].str.strip()
    inpt_OASIS = oasis_cleaner(inpt_OASIS, True, list(inpt_agenda['Year'].unique()))

    if breaking == 1:
        print('Returning "inpt_OASIS" and "inpt_agenda" after phase 1 cleaning.')
        return {"inpt_agenda" : inpt_agenda, "inpt_OASIS": inpt_OASIS}

    #phase 2: Initial matching
    df = pd.merge(inpt_OASIS, inpt_agenda, on=['Organization Name', 'Year'], how='right') #initial match

    if breaking == 2:
        print('Returning "df" after merging "inpt_OASIS" and "inpt_agenda" in phase 2.')
        return {"df" : df}

    #phase 3: cleaning columns
    if valid_cols is None: 
        #standard settings is to use the standard column layout
        standard_ficomm_layout = ['Org ID', 'Organization Name', 'Org Type', 'Callink Page', 'OASIS RSO Designation', 'OASIS Center Advisor', 'Year', 'Year Rank', 'Active', 'Ficomm Meeting Date', 'Ficomm Decision', 'Amount Allocated']
        df = df[standard_ficomm_layout]
    else: 
        assert is_type(valid_cols, str), "Inputted 'valid_cols' not strings or list of strings."
        assert in_df(valid_cols, df), "Inputted'valid_cols' not detected in df."
        df = df[valid_cols]

    if breaking == 3:
        print('Returning "df" after cleaning columns in phase 3.')
        return {"df" : df}

    #phase 4(O): apply fuzzywuzzy/nlp name matcher for names that are slightly mispelled
    if close_matching:
        if custom_close_match_settings is None:
            assert in_df(['Organization Name', 'OASIS RSO Designation'], inpt_OASIS)

            updated_df, _ = close_match_sower(df, inpt_OASIS, 'Organization Name', 'OASIS RSO Designation', 87.9,  sa_filter) #optimal settings based on empirical testing
            
            if breaking == 3.5:
                print('Returning "updated_df" immediately after creation in the middle of phase 4.')
                return {"updated_df" : updated_df}

            updated_df = asuc_processor(updated_df)
            failed_match = updated_df[updated_df['OASIS RSO Designation'].isna()]['Organization Name']
            print(f"Note some club names were not recognized: {failed_match}")
        else: 
            updated_df, _ = close_match_sower(df, inpt_OASIS, *custom_close_match_settings) #optimal settings based on empirical testing
            
            if breaking == 3.5:
                print('Returning "updated_df" immediately after creation in the middle of phase 4.')
                return {"updated_df" : updated_df}
            
            updated_df = asuc_processor(updated_df)
            failed_match = updated_df[updated_df['OASIS RSO Designation'].isna()]['Organization Name']
            print(f"Note some club names were not recognized: {failed_match}")
    
    if breaking == 4:
        print('Returning "updated_df" and "failed_match" after matching names with OASIS in phase 4')
        return {"updated_df" : updated_df, "failed_match" : failed_match}
    
    #phase 5: meeting number
    Ficomm23234_meeting_number = {updated_df['Ficomm Meeting Date'].unique()[i] : i + 1 for i in range(len(updated_df['Ficomm Meeting Date'].unique()))}
    updated_df['Meeting Number'] = updated_df['Ficomm Meeting Date'].map(Ficomm23234_meeting_number)

    #phase 6: re-cleaning numbers in "Ficomm Decision" column
    updated_df['Ficomm Decision'] = updated_df['Ficomm Decision'].apply(lambda s: 'Approved' if s.isdigit() else s)
    
    #phase 7: approved only df
    approved = updated_df[updated_df['Amount Allocated'] > 0]

    #phase 8: FR processing
    # FR_Processor
    
    return approved, updated_df

# def Join_OASIS(df, cleaned_OASISdf, left_on, right_on, right_keep=['OASIS RSO Designation']):
    
