import numpy as np
import pandas as pd
import spacy
nlp_model = spacy.load("en_core_web_md")
from sklearn.metrics.pairwise import cosine_similarity 
from rapidfuzz import fuzz, process

def sa_filter(entry):
        """
        Splits the entry into two parts (before and after "Student Association") for fuzzy matching,
        while retaining the full name for the final output.
        
        Parameters:
        - entry (str): The original club name to be processed.
        
        Returns:
        - tuple: (filtered_name, full_name, filter_applied)
        - If there is no relevant filtered name (ie filtered was not applied), filtered_name is False

        Version 1.0
        - Maybe make it regex to handle names like 'Student Association of Data Science' cuz then it extracts 'of data science' and lower cases it
        """
        parts = entry.lower().split("student association")
        filter_applied = False
        if len(parts) > 1:
            before = parts[0].strip()  # Text before "Student Association"
            after = parts[1].strip()  # Text after "Student Association"
            # Concatenate the simplified name for matching (without "Student Association")
            filtered_name = before + " " + after
            filter_applied = True
        else:
            filtered_name = entry  # No "Student Association", use the full name for matching
        
        return entry, filtered_name, filter_applied

def close_match_sower(df1, df2, matching_col, mismatch_col, fuzz_threshold, filter = None, nlp_processing = False, nlp_process_threshold = None, nlp_threshold = None):
    """
    Matches rows in df1 to df2 based on fuzzy matching and optional NLP embedding similarity.

    Parameters:
    - df1 (pd.DataFrame): Primary dataframe with unmatched entries. Has already been merged once and has some NaN rows. 
    - df2 (pd.DataFrame): Secondary dataframe with potential matches.
    - matching_col (str): Column for matching on in both dataframes (e.g., "Organization Name").
    - mismatch_col (str): Column in df1 that shows up as NaN for unmatched rows (e.g., "Amount Allocated").
    - filter (func): Takes in a filtering function to be applied to individual club names. NOTE the function MUST return 3 outputs for name, processing_name, filt_applied respectively. 
    - fuzz_threshold (int): EXCLUSIVE Minimum score for accepting a fuzzy match (0-100).
    - nlp_processing (bool): Toggle NLP-based matching; default is False.
    - nlp_process_threshold (float, optional): EXCLUSIVE Minimum fuzzy score to attempt NLP-based matching.
    - nlp_threshold (float, optional): EXCLUSIVE Minimum cosine similarity score for accepting an NLP match.

    Returns:
    - pd.DataFrame: Updated dataframe with new matches filled from df2.
    - list: List of tuples containing unmatched entries with reasons.

    Version 2.2
    - Maybe also make sure filter is applied to df2[matching_col] cuz if we apply the filter to the names we're tryna match they should also be applied to the name list we're matching against
    otherwise you're obviously gonna have a hard time matching things like Pakistani to Pakistani Student Assoication. 
    Oh actually this is a catch 22, if you have a Pakistani Student Association and a Pakastani Engineers Association you might not want to filter out "Student Association"
    But if you have "Pakistani Student Association" vs "Kazakstani Student Association" then you do need a filter. 

    Changelog:
    - Made nlp processing toggleable for more precise testing (ver 2.2)
    - Added `nlp_process_threshold` to minimize unnecessary NLP comparisons. (ver 2.1)
    - Improved efficiency by applying the NLP model only to rows with scores below `fuzz_threshold`. (ver 2.0)
    - Enhanced error handling for unmatched cases. (ver 1.1)
    """
    
    assert isinstance(fuzz_threshold, (float, int)), "fuzz_threshold must be an integer."
    if nlp_processing:
        assert isinstance(nlp_process_threshold, (float, int)), "nlp_process_threshold must be a float or int."
        assert isinstance(nlp_threshold, (float, int)), "nlp_threshold must be a float or int."
    
    #isolate entries without a match
    NaN_types = df1[df1[mismatch_col].isna()]
    copy = df1.copy()
    
    #iterate through all entries without a match, searching through df2, identifying closest match, then matching closest match from df2 onto df1
    could_not_match = []
    
    for ind in NaN_types.index:
        if filter is not None:
            name, processing_name, filt_applied = filter(NaN_types.loc[ind, matching_col])
            if filt_applied:
                filt_msg = f'Filter applied to processing name {processing_name}'                   
            else: 
                filt_msg = 'Filter not applied'
        else: 
            name = NaN_types.loc[ind, matching_col]
            processing_name = name
            filt_applied = False
            filt_msg = 'No filter inputted'

        match, score, index = process.extractOne(processing_name, df2[matching_col].tolist())

        if score > fuzz_threshold:
            for col in df2.columns: #ensures all info from the relevant row in copy is overwrited with the corresponding info from df2s
                copy.loc[ind, col] = df2.iloc[index][col]
        elif nlp_processing:             
            if score > nlp_process_threshold:
                    
                    embed = df2[matching_col].apply(nlp_model) #indexes of df2 --> indexes of embed object array for each name

                    name_to_check = np.array([nlp_model(processing_name).vector])
                    embeddings = np.stack(embed.apply(lambda x: x.vector)) #indexes of embed object array --> name vectors array
                    similarities = cosine_similarity(name_to_check, embeddings)
                    best_match_index = similarities.argmax()
                    best_score = similarities[0, best_match_index]
                    
                    if best_score * 100 > nlp_threshold: #cosine_similarity spits out a score from 0 to 1 while nlp_thershold goes from 0 to 100 so it needs to be scaled
                        for col in df2.columns:
                            copy.loc[ind, col] = df2.iloc[best_match_index][col]

                    else: 
                        could_not_match.append( (name, filt_msg, f'closest match: {df2[matching_col].iloc[best_match_index]}', 'nlp elimination', best_score * 100) )
            else: 
                could_not_match.append( (name, filt_msg, f'closest match: {match}', 'fuzz elimination', score) )
        else: 
            could_not_match.append( (name, filt_msg, f'closest match: {match}', 'fuzz elimination', score) )
    
    return copy, could_not_match