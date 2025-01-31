import numpy as np
import pandas as pd

from ASUCExplore.Functions.Utils import heading_finder
from ASUCExplore.Functions.Cleaning import is_type

def ABSA_Processor(df, Cats=None, Drop=None, Add=None):
    """Function to take ABSA CSVs and convert into dataframes.
    Cats happens first then Drop then Add, so you can replace the standard setting with dats then drop"""

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
        def _dropper(instance, dictionary):
            """Removes an instance from either 'Header' or 'No Header'."""
            if instance in set(dictionary['Header']): #convert to set for amortized O(1) membership checking, yay hashsets
                dictionary['Header'].remove(instance)
            elif instance in set(Types['No Header']):
                dictionary['No Header'].remove(instance)
            else: 
                raise ValueError(f"""Drop input {instance} not in any of the subframes set to be selected. Subframes to be selected include:
                                    'Header' subframes: {Types['Header']}
                                    'No Header' subframes: {Types['No Header']}
                                """)
            
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
        header_result = heading_finder(df, 0, label, 1, 'SUBTOTAL', 'exact', 'contains')
        header_result['Org Category'] = np.full(len(header_result), label)
        header_result = header_result.loc[:, ~header_result.columns.isna()]
        header_result = header_result.reset_index(drop=True)
        sub_frames.append(header_result)
    for label in Types['No Header']:
        no_header_result = heading_finder(df, 0, label, 0, 'SUBTOTAL', 'exact', 'contains') #if u don't include zero it interprets 'SUBTOTAL' as shift param
        no_header_result['Org Category'] = np.full(len(no_header_result), label)
        no_header_result = no_header_result.loc[:, ~no_header_result.columns.isna()]
        no_header_result = no_header_result.reset_index(drop=True)
        no_header_result.columns = no_header_result.columns.str.strip()
        if label in no_header_result.columns:
            no_header_result = no_header_result.rename(columns={label: 'Organization'})
        else:
            print(f"Warning: Column '{label}' not found in DataFrame columns. Available columns: {no_header_result.columns.tolist()}")
        sub_frames.append(no_header_result)
    
    return pd.concat(sub_frames, ignore_index=True)
