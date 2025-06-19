import numpy as np
import pandas as pd

from ASUCExplore.Cleaning import in_df, is_type
from ASUCExplore.Utils import column_converter, heading_finder, column_renamer

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

def asuc_processor(df):
   """
   Checks for any ASUC orgs in a df and updates those entries with the 'ASUC' label.
   Developed cuz ASUC orgs aren't on OASIS so whenever they apply for ficomm funds and their names show up, it shows up as "NA" club type.
   """
   def asuc_processor_helper(org_name):
      asuc_exec = set(['executive vice president', 'office of the president', 'academic affairs vice president', 'external affairs vice president', 'student advocate']) #executive vice president is unique enough, just president is not
      asuc_chartered = set(['grants and scholarships foundation', 'innovative design', 'superb']) #incomplete: address handling extra characters (eg. grants vs grant)
      asuc_commission = set(['mental health commision', 'disabled students commission', 'sustainability commission', 'sexual violence commission']) #incomplete
      asuc_appointed = set(['chief finance officer', 'chief communications officer', 'chief legal officer', 'chief personel officer', 'chief technology officer']) 
      if 'senator' in org_name.lower() and 'asuc' in org_name.lower():
         return 'ASUC: Senator'
      elif org_name.lower() in asuc_exec and org_name.lower().contains('asuc'):
         return 'ASUC: Executive'
      elif org_name.lower() in asuc_commission and org_name.lower().contains('asuc'):
         return 'ASUC: Commission'
      elif org_name.lower() in asuc_chartered: # I'm not sure if chartered programs put 'ASUC' in their shit?
         return 'ASUC: Chartered Program'
      elif org_name.lower() in asuc_appointed and org_name.lower().contains('asuc'):
         return 'ASUC: Appointed Office'
      else:
         return org_name
      

   assert in_df(['Organization Name', 'OASIS RSO Designation'], df), f'"Organization Name" and/or "OASIS RSO Designation" not present in inputted df, both must be present but columns are {df.columns}.'
   if not is_type(df['Organization Name'], str):
      df = column_converter(df, 'Organization Name', str)
      
   if not is_type(df['OASIS RSO Designation'], str):
      df = column_converter(df, 'OASIS RSO Designation', str)

   cleaned = df.copy()

   cleaned['OASIS RSO Designation'] = cleaned['Organization Name'].apply(asuc_processor_helper)
   cleaned['ASUC'] = cleaned['OASIS RSO Designation'].apply(lambda x: 1 if 'ASUC' in x else 0)

   return cleaned

def OASIS_Processor(df, year, rename=None, col_types=None, existing=None):
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
        column_converter(cleaned_df, OClean_Int_Cols, int, mutate = True)
        column_converter(cleaned_df, OClean_Str_Cols, str, mutate = True)
    else:
        #expecting col_types to be 
        for key in col_types.keys(): 
            column_converter(cleaned_df, col_types[key], key, mutate = True)
    
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
        cleaned_df = pd.concat([cleaned_df, existing]).sort_values(by=['Year Rank', 'Organization Name'])
        return cleaned_df
    else: 
        return cleaned_df
    
def OASIS_Abridged(df, rename=None, col_types=None, existing=None):
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
    
    if col_types is None: #phase 4
        OClean_Str_Cols = ['Org ID', 'Organization Name', 'Org Type', 'OASIS RSO Designation', 'OASIS Center Advisor']
        cleaned_df = column_converter(cleaned_df, OClean_Str_Cols, str, mutate = False)
    else:
        #expecting col_types to be 
        for key in col_types.keys(): 
            column_converter(cleaned_df, col_types[key], key, mutate = True)
    
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
        cleaned_df = pd.concat([cleaned_df, existing]).sort_values(by=['Year Rank', 'Organization Name'])
        return cleaned_df
    else: 
        return cleaned_df

