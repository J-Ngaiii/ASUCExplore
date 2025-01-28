import numpy as np
import pandas as pd

from Utils import heading_finder

def ABSA_Processor(df, Types=None):
    """Function to take ABSA CSVs and convert into dataframes."""

    if Types is None:
        Types = {}
        Types['Header'] = ['ASUC Chartered Programs and Commissions', 'Publications', 'Student Activity Groups', 'Student-Initiated Service Group']
        Types['No Header'] = ['Office of the President', 'Office of the Executive Vice President', 'Office of External Affairs Vice President', 'Office of the Academic Affairs Vice President', "Student Advocate's Office", 'Senate', 'Appointed Officials', 'Operations', 'Elections', 'External Expenditures']
        Types['Final Counts'] = ['ASUC External Budget', 'ASUC Internal Budget', 'FY25 GENERAL BUDGET']
    
    sub_frames = []
    for label in Types['Header']:
        try: 
            sub_frames.append(heading_finder(df, 1, label, 1, 'SUBTOTAL', 'contains'))
        except Exception as e: 
            raise e
        
    for label in Types['No Header']:
        try: 
            sub_frames.append(heading_finder(df, 1, label, 'SUBTOTAL', 'contains'))
        except Exception as e: 
            raise e
    
    return pd.concat(sub_frames)
