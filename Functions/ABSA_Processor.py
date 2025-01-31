import numpy as np
import pandas as pd

from ASUCExplore.Functions.Utils import heading_finder

def ABSA_Processor(df, Types=None):
    """Function to take ABSA CSVs and convert into dataframes."""

    if Types is None:
        Types = {}
        Types['Header'] = ['ASUC Chartered Programs and Commissions', 'Publications (PUB) Registered Student Organizations', 'Student Activity Groups (SAG)', 'Student-Initiated Service Group (SISG)']
        Types['No Header'] = ['Office of the President', 'Office of the Executive Vice President', 'Office of External Affairs Vice President', 'Office of the Academic Affairs Vice President', "Student Advocate's Office", 'Senate', 'Appointed Officials', 'Operations', 'Elections', 'External Expenditures']
        Types['Final Counts'] = ['ASUC External Budget', 'ASUC Internal Budget', 'FY25 GENERAL BUDGET']
    
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
