import numpy as np
import pandas as pd
import re
import spacy
nlp_model = spacy.load("en_core_web_md")
from sklearn.metrics.pairwise import cosine_similarity 
from rapidfuzz import fuzz, process

from ASUCExplore.Cleaning import in_df, is_type
from ASUCExplore.Utils import column_converter

def _find_chunk_pattern(starts, ends, end_prepattern = '\d\.\s'):
      """
      Extracts a chunk of text from 'inpt' text based on start and end keywords.
      starts (list[str]): List of keywords to start the chunk of text we want to extract
      ends (list[str]): List of keywords to end the chunk of text we want to extract
      end_prepattern (str): Regex pattern to append before every end keyword
         - for example meeting agendas are structure 1. Contingency, 2. Sponsorship, etc so if you want the chunk to end at 'Sponsorship' you add in a ending prepattern to catch the '2. '.
      
      example pattern to constryct: Contingency\s*([\s\S]*)(?:\d\.\sAdjournment|\d\.\sSponsorship)
      
      """
      assert len(starts) != 0, 'starts is an empty list'
      assert is_type(starts, str), 'starts is not a list of strings'
      assert len(ends) != 0, 'ends is an empty list'
      assert is_type(ends, str), 'ends is not a list of strings'

      assert isinstance(end_prepattern, str), f'end_prepattern should be a string but is type {type(end_prepattern)}'
      
      pattern = ''
      if len(starts) == 1:
          pattern += starts[0]
      else: 
         pattern = '(?:'
         for start_keyword in starts[:-1]: 
            pattern += start_keyword + '|'
         pattern += starts[-1] + ')'
      
      pattern += '\s*?([\s\S]*?)(?:' # make sure to have the '*?' to do non-greedy matching

      if len(ends) == 1:
         pattern += ends[0]
      else: 
         for end_keyword in ends[:-1]: 
            pattern += end_prepattern + end_keyword + '|'
         pattern += end_prepattern + ends[-1]

      pattern += ')'
      return pattern

def _motion_processor(club_names, names_and_motions):
   """Takes in a list of club names for a given chunk and a list of club names and motions. Outputs a dictionary of club names mapped to keys containing the relevant motion. 
   club_names (list[str]): List of club names (eg. ['V-Day at Berkeley', 'Aion', 'Volunteer Income Tax Assistance Program', 'ASUC Menstrual Equity Commission', 'ASUC Menstrual Equity Commission'])
   names_and_motions (list[str]): List of club names and motions (eg. ['V-Day at Berkeley', 'Motion to approve $400 by Senator Manzoor', 'Seconded by Senator Ponna', 'Aion', 'Motion to approve $300 by Senator Manzoor ', 'Seconded by Senator Ponna ')
   """
   # print(f"Club Names: {club_names}")
   # print(f"Club Motions: {names_and_motions}")
   rv = {}
   repeats = dict(zip(club_names, [0]*len(club_names)))
   club_set = set(club_names)  
   curr_club = None
   for curr in names_and_motions: 
      if curr in club_set: 
         if curr in rv: #to register clubs that get repeated in the agenda due to multiple submissions
            curr_club = curr + f" ({str(repeats[curr] + 1)})"
         else:
            curr_club = curr
         rv[curr_club] = [] #to register clubs with no motions
      else: 
         if curr_club is None:
            print(f"""WARNING line skip occured with line: {curr}
            total list is: {names_and_motions}""")
         else:
            rv[curr_club].append(curr)

   return rv

def Agenda_Processor(inpt, start=['Contingency Funding', 'Contingency'], end=['Finance Rule', 'Rule Waiver', 'Space Reservation', 'Sponsorship', 'Adjournment', 'ABSA', 'ABSA Appeals'], identifier='(\w+\s\d{1,2}\w*,\s\d{4})'):
   """
   You have a chunk of text from the document you want to turn into a table and an identifier for that chunk of text (eg. just the Contingency Funding section and the identifeir is the date). 
   Thus function extracts the chunk and converts it into a tabular format.

   input (str): The raw text of the agenda to be processed. Usually a .txt file
   identifier (str): Regex pattern to extract a certain piece of text from inpt as the identifier for the chunk extracted from inpt
   """
   date = re.findall(rf"{identifier}", inpt)[0]
   pattern = _find_chunk_pattern(start, end)
   # print(f"Pattern: {pattern}")
   chunk = re.findall(rf"{pattern}", inpt)[0]

   # print(f"chunk: {chunk}")

   valid_name_chars = '\w\s\-\_\*\&\%\$\+\#\@\!\(\)\,\'\"' #seems to perform better with explicit handling for special characters? eg. for 'Telegraph+' we add the plus sign so regex will pick it up
   club_name_pattern = f'\d+\.\s(?!Motion|Seconded)([{valid_name_chars}]+)\n(?=\s+\n|\s+\d\.)' #first part looks for a date, then excluding motion and seconded, then club names
   club_names = list(re.findall(club_name_pattern, chunk)) #just matches club names --> list of tuples of club names

   names_and_motions = list(re.findall(rf'\d+\.\s(.+)\n?', chunk)) #pattern matches every single line that comes in the format "<digit>.<space><anything>""
   motion_dict = _motion_processor(club_names, names_and_motions)
   # print(f"Motion Dict: {motion_dict}")
   

   decisions = []
   allocations = []
   for name in motion_dict.keys():
      
      if motion_dict[name] == []:
         decisions.append('No record on input doc')
         allocations.append(np.nan)
         
      else:
         sub_motions = " ".join(motion_dict[name]) #flattens list of string motions into one massive continuous string containing all motions
         # print(f'sub-motions: {sub_motions}')

         #for handling multiple conflicting motions (which shouldn't even happen) we record rejections > temporary tabling > approvals > no input
         #when in doubt assume rejection
         #check if application was denied or tabled indefinetly
         if re.findall(r'(tabled?\sindefinetly)|(tabled?\sindefinitely)|(deny)', sub_motions) != []: 
            decisions.append('Denied or Tabled Indefinetly')
            allocations.append(0)
         #check if the application was tabled
         elif re.findall(r'(tabled?\suntil)|(tabled?\sfor)', sub_motions) != []:
            decisions.append('Tabled')
            allocations.append(0)
         #check if application was approved and for how much
         elif re.findall(r'[aA]pprove', sub_motions) != []:
            dollar_amount = re.findall(r'[aA]pprove\s(?:for\s)?\$?(\d+)', sub_motions)
            if dollar_amount != []:
               decisions.append('Approved')
               allocations.append(dollar_amount[0])
            else:
               decisions.append('Approved but dollar amount not listed')
               allocations.append(np.nan) # not listed appends NaN
         #check if there was no entry on ficomm's decision for a club (sometimes happens due to record keeping errors)
         elif sub_motions == '':
            decisions.append('No record on input doc')
            allocations.append(np.nan)
         else:
            decisions.append('ERROR could not find conclusive motion')
            allocations.append(np.nan)

   rv = pd.DataFrame({
      'Organization Name' : pd.Series(motion_dict.keys()).str.strip(), #solves issue of '\r' staying at the end of club names and messing things up
      'Ficomm Decision' : decisions, 
      'Amount Allocated' : allocations
      }
   )
   # print(f"Agenda Processor Final df: {rv}")

   return rv, date

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
   if ~is_type(df['Organization Name'], str):
      column_converter(df, 'Organization Name', str)
      
   if ~is_type(df['OASIS RSO Designation'], str):
      column_converter(df, 'OASIS RSO Designation', str)

   cleaned = df.copy()

   cleaned['OASIS RSO Designation'] = cleaned['Organization Name'].apply(asuc_processor_helper)
   cleaned['ASUC'] = cleaned['OASIS RSO Designation'].apply(lambda x: 1 if 'ASUC' in x else 0)

   return cleaned
