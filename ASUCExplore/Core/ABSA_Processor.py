import numpy as np
import pandas as pd

# full path should be the way as long as: 

from ASUCExplore.Utils import heading_finder
from ASUCExplore.Cleaning import is_type

def _dropper(instance, dictionary):
    """
    Function Removes an occurence of 'instance' from either 'Header' or 'No Header'.
    """
    if instance in set(dictionary['Header']): #convert to set for amortized O(1) membership checking, yay hashsets
        dictionary['Header'].remove(instance)
    elif instance in set(dictionary['No Header']):
        dictionary['No Header'].remove(instance)
    else: 
        raise ValueError(f"""Drop input {instance} not in any of the subframes set to be selected. Subframes to be selected include:
                            'Header' subframes: {dictionary['Header']}
                            'No Header' subframes: {dictionary['No Header']}
                        """)

def ABSA_Processor(df: pd.DataFrame, Cats: dict[str, list[str]] = None, Drop: str = None, Add: str = None) -> pd.DataFrame:
    """
    Function to take ABSA CSVs and convert into dataframes.
    Cleaning Process: Cats happens first then Drop then Add, so you can replace the standard setting with dats then drop
    Cats (dict) : dictionary with keys 'Header', 'No Header' and 'Final Counts' and values containing lists of column names. 
        The function then uses Cats to determine if it should handle for the presence or absence of a header when creating sub-dataframes for each section of interest.
        Once all sub-frames are created (eg. one for RSOs, one for ASUC President), they get appended together.
    """

    Types = {
        'Header': [
            'ASUC Chartered Programs and Commissions', 'Publications (PUB) Registered Student Organizations',
            'Student Activity Groups (SAG)', 'Student-Initiated Service Group (SISG)'
        ],
        'No Header': [
            'Office of the President', 'Office of the Executive Vice President', 'Office of External Affairs Vice President',
            'Office of the Academic Affairs Vice President', "Student Advocate's Office", 'Senate', 'Appointed Officials',
            'Operations', 'Elections', 'External Expenditures'
        ],
        'Final Counts': ['ASUC External Budget', 'ASUC Internal Budget', 'FY25 GENERAL BUDGET'] #may be referenced later if we build out the function further
    } 

    if Cats is not None:
        assert isinstance(Cats, dict), "Cats must be a dictionary."
        if not all(key in Cats.keys() for key in ['Header', 'No Header']):
            raise ValueError("Cats must specify both 'Header' and 'No Header' categories.") 
        Types['Header'] = Cats['Header']
        Types['No Header'] = Cats['No Header']

    if Drop is not None:
        assert is_type(Drop, str), 'Drop must be a string or iterable of strings specifying column type'
        if isinstance(Drop, str):
            _dropper(Drop, Types)
        else:
            for cat in Drop: #convert to set for amortized O(1) membership checking, yay hashsets
                _dropper(cat, Types)

    # if Add is not none: 
        ### TO DO ###


    sub_frames = []
    for label in Types['Header']:
        header_result: pd.DataFrame = heading_finder(df = df, start_col = 0, start = label, shift = 1, end = 'SUBTOTAL', start_logic = 'exact', end_logic = 'contains')
        if header_result.empty: 
            print(f"Warning: No data found for label {label} under the 'Header' category.")
        else: 
            header_result['Org Category'] = np.full(len(header_result), label)
            header_result = header_result.loc[:, ~header_result.columns.isna()] # drop any null columns
            header_result = header_result.reset_index(drop=True)
            sub_frames.append(header_result)
    for label in Types['No Header']:
        no_header_result: pd.DataFrame = heading_finder(df = df, start_col = 0, start = label, shift = 0, end = 'SUBTOTAL', start_logic = 'exact', end_logic = 'contains') 
        if no_header_result.empty: 
            print(f"Warning: No data found for label {label} under the 'No Header' category.")
        else:
            no_header_result['Org Category'] = np.full(len(no_header_result), label)
            no_header_result = no_header_result.loc[:, ~no_header_result.columns.isna()] # drop any null columns
            no_header_result = no_header_result.reset_index(drop=True)
            no_header_result.columns = no_header_result.columns.str.strip()
            if label in no_header_result.columns:
                no_header_result = no_header_result.rename(columns={label: 'Organization'})
            else:
                print(f"Warning: Column '{label}' not found in DataFrame columns. Available columns: {no_header_result.columns.tolist()}")
            sub_frames.append(no_header_result)
            
    # for label in Types['Final Counts']:
    #     result = df[df.iloc[:,0] == label]
    #     if result.empty:
    #         raise ValueError(f'No exact matches found in first column for label {label}')
        ### TO DO ###

    return pd.concat(sub_frames, ignore_index=True)
