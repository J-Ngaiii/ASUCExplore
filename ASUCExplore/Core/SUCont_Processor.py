import pandas as pd
from ASUCExplore.Utils import column_converter

def SU_Cont_Processor(df, str_cols=None, date_cols=None, float_cols=None):
    """
    Expected Intake: Df with following columns: 
    """
    SUContCleaned = df.copy()

    #Phase 1: column conversions
    if str_cols is None:
        str_cols = ['Account Name',
                    'Account Description',
                    'Transaction Reference #',
                    'Reconciled',
                    'Created By',
                    'Payee/Source First Name',
                    'Payee/Source Last Name',
                    'Originator Account Name',
                    'From Request - Account Name',
                    'Request Number',
                    'From Request - Subject',
                    'From Request - Payee First Name',
                    'From Request - Payee Last Name',
                    'From Request - Payee Address1',
                    'From Request - Payee Address2',
                    'From Request - Payee City',
                    'From Request - Payee State',
                    'From Request - Payee ZIP',
                    'Memo',
                    'Category',
                    'Type',
                    'Transaction Method']
    if float_cols is None:
        float_cols = ['Amount', 'Ending Balance After', 'Available Balance After']
    if date_cols is None:
        date_cols = ['Date']

    column_converter(SUContCleaned, str_cols, str)
    column_converter(SUContCleaned, date_cols, pd.Timestamp)
    column_converter(SUContCleaned, float_cols, float)

    #Phase 2: cleaning out dollar signs
    SUContCleaned[[
    'Amount',
    'Ending Balance After', 
    'Available Balance After'
    ]] = SUContCleaned[[
    'Amount',
    'Ending Balance After', 
    'Available Balance After'
    ]].apply(lambda col: col.str.replace('[\$,]', '', regex=True).astype(float))

    #Phase 3: adding admin category
    SUContCleaned['Admin'] = SUContCleaned['Category'].apply(lambda x: 1 if ('Admin use only') in x else 0)

    #Phase 4: adding recipient column for ASUC (NOT GENERALIZED)
    SUContCleaned['Recipient'] = SUContCleaned['Memo'].str.extract(r'[FR|SR]\s\d{2}\/\d{2}[-\s](?:[F|S]\d{2}\s-\s|\d+\s)(.+)') #isolating recipient from `memo` column, only processes FR memos
    SUContCleaned['Recipient'] = SUContCleaned['Recipient'].apply(lambda x: 'ASUC ' + x if (type(x) is str) and ('Office of Senator' in x) else x) #make sure type check goes first or else function tries to check membership against NaN values which are floats
    SUContCleaned['Recipient'] = SUContCleaned['Recipient'].apply(lambda x: 'ASUC - Office of the Executive Vice President' if (type(x) is str) and (x == 'ASUC EVP') else x) #this is only here cuz one time EVP was entered as "EVP" rather than the full name

    return SUContCleaned