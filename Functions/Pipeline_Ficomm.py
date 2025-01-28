import numpy as np
import pandas as pd
import re
import spacy
nlp_model = spacy.load("en_core_web_md")
from sklearn.metrics.pairwise import cosine_similarity 
from rapidfuzz import fuzz, process

from .Cleaning import in_df, is_type

def _cont_approval_helper(input, start=['Contingency Funding'], end=['Finance Rule', 'Space Reservation', 'Sponsorship']):
   """
   Extracts and organizes data from a given agenda string, sorting it into a dictionary where:
   - Keys are the meeting dates.
   - Values are sub-dictionaries with club names as keys and Ficomm decisions (such as approval amounts or tabled status) as values.

   The function searches for the specified start and end keywords to define the boundaries of the "Contingency Funding" section within the input agenda. It processes each section for clubs and their respective motions, and returns a structured dictionary containing the results.

   Version 3.2: Changes with ficomm meeting date being pd.Timestamp object and denied applications having "0" for amount allocated.
   - Handling multiple conflicting motions (which shouldn't even happen) currently is done by prioritizing record rejections > temporary tabling > approvals > no input. Maybe we change this later down the line.

   Args:
      input (str): The raw text of the agenda to be processed.
      start (list, optional): A list of keywords that mark the beginning of the section to extract (default is `['Contingency Funding']`).
      end (list, optional): A list of keywords that mark the end of the section to extract (default is `['Finance Rule', 'Space Reservation', 'Sponsorship']`).

   Returns:
      dict: A dictionary where each date (str) is a key, and the value is another dictionary mapping club names (str) to Ficomm decisions (str).

   Raises:
      Exception: If the input text is empty.
      Exception: If the start or end lists are empty or not valid lists.

   Notes:
      - The function uses regular expressions to extract meeting dates, clubs, and motions.
      - Efficiency may not be optimal for large agendas due to the regex-based search.
      - This function is designed for agendas in a specific format and may not work for other types of input.
   """
   
   def _cont_appender_helper(pattern, start_list, end_list):
      """
      Constructs a regular expression pattern to extract text between specified start and end keywords.

      This helper function is designed to generate a regex pattern that can be used to extract content from a text document between the first occurrence of any keyword in the `start_list` and the first occurrence of any keyword in the `end_list`.

      It is primarily used for processing Ficomm agendas, where sections are delineated by specific keywords such as 'Contingency Funding' and 'Finance Rule'. The generated regex pattern is designed for use in functions like `re.findall()` to extract sections of text between these markers.

      Args:
         pattern (str): The base regular expression pattern to which the start and end keywords will be appended. This serves as the starting point for building the full regex.
         start_list (list): A list of keywords that mark the beginning of the section to capture. The function will match the first occurrence of any of these keywords.
         end_list (list): A list of keywords that mark the end of the section to capture. The function will stop capturing when any of these keywords is found.

      Returns:
         str: A complete regular expression pattern that matches content between the start and end keywords. The pattern can be used in a function like `re.findall()` to extract the desired sections from the input text.

      Raises:
         Exception: If either `start_list` or `end_list` is empty or not a list.

      Notes:
         - This function uses non-greedy matching and lookahead assertions to ensure that the text between the start and end markers is captured accurately.
         - The returned pattern can be used to capture multiple sections if they occur in the input text. It is specifically designed for processing Ficomm agenda formats but could be adapted for other purposes.
      
      Example:
         >>> _cont_appender_helper(pattern, ['Contingency Funding', 'Space Reservation'], ['Sponsorship', 'Rule Waiver'])
         Appender will return a regex pattern that when fed into an extraction function like re.findall returns all text that comes
         after the first appearance of 'Contingency Funding' or 'Space Reservation' and before the first appearance of 'Sponsorship' or 'Rule Waiver'
      """

      #example input: '{d}[\s\S]*?('
      #example output: '{d}[\s\S]*?(Contingency Funding[\s\S]*?(?=(?:Finance Rule|Space Reservation|Sponsorship|$)))'
      
      if (type(start_list) is not list):
         raise Exception('_cont_appender_helper start_list argument is not list')
      elif len(start_list) == 0:
         raise Exception('_cont_appender_helper start_list argument is empty')
      elif len(start_list) == 1:
          pattern += start_list[0]
      else: 
         for start_keyword in start_list[:-1]: 
            pattern += start_keyword + '|'
         pattern += start_list[-1]
      
      pattern += '[\s\S]*?(?=(?:'

      if (type(end_list) is not list):
         raise Exception('_cont_appender_helper end_list argument is not list')
      elif len(end_list) == 0:
         raise Exception('_cont_appender_helper end_list argument is empty')
      elif len(end_list) == 1:
         pattern += end_list[0]
      else: 
         for end_keyword in end_list[:-1]: 
            pattern += end_keyword + '|'
         pattern += end_list[-1]
      
      pattern += '|$)))'
      return pattern



   if input == "":
      raise Exception('Input text is empty')
   
   Dates = re.findall(r'(\w+\s\d{1,2}\w*,\s\d{4})', input, re.S)
   Dates_Dict = {}
   #extract contingency chunks

   #looks for 'Contingency Funding' phrase to signal beginning of contingency fund applications section
   #looks for 'Finance Rule', 'Space Reservation' or 'Sponsorship' to signal end of contingency fund applications section
   
   for d in Dates:
      Dates_Dict[d] = None
      initial_pattern = f'{d}[\s\S]*?('
      final_pattern = _cont_appender_helper(initial_pattern, start, end)
      match = re.findall(rf'{final_pattern}', input)

      #if we get a chunk of contingency apps under this meeting date
      if match != []:
         chunk = match[0]

         #club name pattern works by checking for a digit, (all characters of a club name) in capture group to be returned, then a new line, spaces and the next digit signifiying the start of the first motion
         #NOTE for club names with no motions like "3. No name club <new line with no text>, <new line> 2. " it matches the empty lines till "2."
         valid_name_chars = '\w\s\-\_\*\&\%\$\#\@\!\(\)\,\'\"'
         club_names = re.findall(rf'\d+\.\s([{valid_name_chars}]+)\n(?=\s+\n|\s+\d\.)', chunk) #just matches club names --> list of tuples of club names
         
         #motions pattern handles text of the form (this is the same as what is outputted with chunk): 
         #Contingency Funding (whatever starting keyword, as long as there's no number before it)
            # <number>. <club name> 
               #<number>. <Motioning statement> 
         #can handle capturing multiple motioning statements that start with 'Motion ', 'Unanimously ' or 'Senator '
         #can names with dashes or asterisks in between like 'MEMSSA Ad-Hoc Committee *'
         #should in theory be able to handel special characters: -, _, *, &, $, #, @, !, (, ,), <commas>, ", '
         #NOTE DO NOT TRY TO HANDLE CLUB NAMES WITH PERIODS it bricks club name's abolity to match
         #CANNOT handle tabs rather than new lines infront of club names
         motions = re.findall(rf'\d+\.\s([{valid_name_chars}]+)\n\s*((?:\s+\d+\.\s(?:Motion[^\n]*?|Unanimously[^\n]*?|Senator[^\n]*?)[.!?](?:\s+Seconded[^\n]*?[.!?])?(?:\s+(?:Motion\spassed|Passed\sby)[^\n]*?[.!?])?)+)', chunk) #matches motions IF there's club names --> list of tuples of (club name, motions)
         if motions == [] and club_names == []:
            Dates_Dict[d] = 'Section starting and ending with desired keywords detected but no motions or club in valid formatting detected'
         else: 
            clubs_with_motions = { t[0]:t[1].strip() for t in motions} # iterates thru the (club name, motions) tuples
            Entries = {}
            #must iterate through cuz we want to note down the names of clubs with no motions
            for name in club_names:
               
               if name not in clubs_with_motions.keys():
                  Entries[name] = 'No record on input doc'
                  
               else:
                  sub_motions = clubs_with_motions[name]

                  #for handling multiple conflicting motions (which shouldn't even happen) we record rejections > temporary tabling > approvals > no input
                  #when in doubt assume rejection
                  #check if application was denied or tabled indefinetly
                  if re.findall(r'(tabled?\sindefinetly)|(tabled?\sindefinitely)|(deny)', sub_motions) != []: 
                     Entries[name] = 'Denied or Tabled Indefinetly'
                  #check if the application was tabled
                  elif re.findall(r'(tabled?\suntil)|(tabled?\sfor)', sub_motions) != []:
                     Entries[name] = 'Tabled'
                  #check if application was approved and for how much
                  elif re.findall(r'[aA]pprove', sub_motions) != []:
                     dollar_amount = re.findall(r'[aA]pprove\s(?:for\s)?\$?(\d+)', sub_motions)
                     if dollar_amount != []:
                        Entries[name] = dollar_amount[0]
                     else:
                        Entries[name] = 'Approved but dollar amount not listed'
                  #check if there was no entry on ficomm's decision for a club (sometimes happens due to record keeping errors)
                  elif sub_motions == '':
                     Entries[name] = 'No record on input doc'
                  else:
                     Entries[name] = 'ERROR could not find conclusive motion'
            Dates_Dict[d] = Entries
      else: 
         Dates_Dict[d] = 'No section starting and ending with the desired keywords detected in input doc for this date'
         
   return Dates_Dict

def _cont_approval_dataframe(dict):
   """
   Converts the nested dictionary produced by `_cont_approval_helper` into a Pandas DataFrame.

   This DataFrame organizes the extracted agenda data, with columns for meeting dates, club names, Ficomm decisions, and allocated amounts (if applicable).

   Args:
      dict (dict): A dictionary containing the processed agenda data, where each date maps to a sub-dictionary of clubs and their Ficomm decisions.

   Returns:
      pandas.DataFrame: A DataFrame with columns 'Ficomm Meeting Date', 'Organization Name', 'Ficomm Decision', and 'Amount Allocated'.
   
   Notes:
      - The `Amount Allocated` column will contain the parsed dollar amount if available; otherwise, it will be set to `-1` if the value is not a number.
      - This function is designed to work with the dictionary structure returned by `_cont_approval_helper`.
   """
   
   dates_list = []
   club_list = []
   result_list = []
   amt_list = []
   for date in dict:
      for club in dict[date]:
         dates_list.append(pd.Timestamp(date))
         club_list.append(club)
         result_list.append(dict[date][club])
         try: 
            amt_list.append(int(dict[date][club]))
         except Exception as e: 
            amt_list.append(0) #NEW CHANGE: check that tabled and denied applications have 0 as amt allocated
   rv = pd.DataFrame({'Ficomm Meeting Date' : dates_list, 'Organization Name' : club_list, 'Ficomm Decision' : result_list, 'Amount Allocated' : amt_list})
   return rv

def cont_approval(input_txt):
   """
   Processes a raw agenda string and converts it into a Pandas DataFrame with club funding decisions.

   This function combines `_cont_approval_helper` for extracting the data and `_cont_approval_dataframe` to format it into a DataFrame for easier analysis.

   Args:
      input_txt (str): The raw text of the Ficomm agenda to be processed.

   Returns:
      pandas.DataFrame: A DataFrame containing columns for the meeting date, organization name, Ficomm decision, and the amount allocated (if specified).

   Notes:
      - This function provides an easy interface for extracting and formatting agenda data into a DataFrame.
      - It handles agenda sections related to contingency funding and applies custom start/end keyword filters if provided.
   """
   return _cont_approval_dataframe(_cont_approval_helper(input_txt))

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
    """Checks for any ASUC orgs in a df and upadtes those entries with the 'ASUC' label.
    Developed cuz ASUC orgs aren't on OASIS so whenever they apply for ficomm funds and their names show up, it shows up as "NA" club type. """
    def asuc_processor_helper(org_name):
        asuc_exec = set(['executive vice president', 'office of the president', 'academic affairs vice president', 'external affairs vice president', 'student advocate']) #executive vice president is unique enough, just president is not
        asuc_chartered = set(['grants and scholarships foundation', 'innovative design', 'superb']) #incomplete: address handling extra characters (eg. grants vs grant)
        asuc_commission = set(['mental health commision', 'disabled students commission', 'sustainability commission', 'sexual violence commission']) #incomplete
        asuc_appointed = set(['chief finance officer', 'chief communications officer', 'chief legal officer', 'chief personel officer', 'chief technology officer']) 
        if org_name.lower().contains('senator') and org_name.lower().contains('asuc'):
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
        
    
    assert in_df(['Organization Name', 'OASIS RSO Designation'], df), '"Organization Name" and/or "OASIS RSO Designation" not present in inputted df.'
    assert is_type(df['Organization Name'], str), "Column 'Organization Name' exists in inputted def but doesn't contain datatype string objects"
    assert is_type(df['OASIS RSO Designation'], str), "Column 'OASIS RSO Designation' exists in inputted def but doesn't contain datatype string objects"
    
    cleaned = df.copy()

    cleaned['OASIS RSO Designation'] = cleaned['Organization Name'].apply(asuc_processor_helper)
    cleaned['ASUC'] = cleaned['OASIS RSO Designation'].apply(lambda x: 1 if x.contains('ASUC') else 0)

    return cleaned
