import numpy as np
import pandas as pd

def FY_Rez_Generator(df, Alphabet_List, appx=True):

    FR_Sheet_RAW = df.copy()
    first_col = FR_Sheet_RAW.columns[0]
    FR_Sheet_RAW[first_col] = FR_Sheet_RAW[first_col].apply(lambda x: 'NaN' if pd.isna(x) else x)

    if appx:
        boundary = FR_Sheet_RAW[FR_Sheet_RAW[first_col] == 'Appx.'].index[1]
        rang = pd.Series(FR_Sheet_RAW[FR_Sheet_RAW[first_col].isin(Alphabet_List)].index)
        rang = rang[rang.apply(lambda x: x < boundary)]
    else:
        rang = pd.Series(FR_Sheet_RAW[FR_Sheet_RAW[first_col].isin(Alphabet_List)].index)
        
    col_mapper = {}
    col_mapper[first_col] = FR_Sheet_RAW.iloc[rang[0] - 1,:][0]
    for i in range(1, len(FR_Sheet_RAW.columns)):
        col_mapper[FR_Sheet_RAW.columns[i]] = FR_Sheet_RAW.iloc[rang[0] - 1,:][i]

    Sheet = FR_Sheet_RAW.iloc[rang, :]
    Sheet = Sheet.rename(columns=col_mapper)
    
    return Sheet

def FR_Processor(df):
    
    FY24_Alphabet = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'.split()
    FY24_Alphabet.extend(
        'AA AB AC AD AE AF AG AH AI AJ AK AL AM AN AO AP AQ AR AS AT AU AV AW AX AY AZ'.split()
    )
    FY24_Alphabet.extend(
        'BB CC DD EE FF GG HH II JJ KK LL MM NN OO PP QQ RR SS TT UU VV WW XX YY ZZ'.split()
    )
    
    return FY_Rez_Generator(df, FY24_Alphabet)
    