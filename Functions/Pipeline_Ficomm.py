import numpy as np
import pandas as pd
import re
import spacy
nlp_model = spacy.load("en_core_web_md")
from sklearn.metrics.pairwise import cosine_similarity 
from rapidfuzz import fuzz, process

def Cont_Approval_Helper(input, start=['Contingency Funding'], end=['Finance Rule', 'Space Reservation', 'Sponsorship']):
   """
   Extracts and organizes data from a given agenda string, sorting it into a dictionary where:
   - Keys are the meeting dates.
   - Values are sub-dictionaries with club names as keys and Ficomm decisions (such as approval amounts or tabled status) as values.

   The function searches for the specified start and end keywords to define the boundaries of the "Contingency Funding" section within the input agenda. It processes each section for clubs and their respective motions, and returns a structured dictionary containing the results.

   Version 3.0: Changes for efficiency and handling clubs with no motions
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
   
   def cont_appender_helper(pattern, start_list, end_list):
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
         >>> cont_appender_helper(pattern, ['Contingency Funding', 'Space Reservation'], ['Sponsorship', 'Rule Waiver'])
         Appender will return a regex pattern that when fed into an extraction function like re.findall returns all text that comes
         after the first appearance of 'Contingency Funding' or 'Space Reservation' and before the first appearance of 'Sponsorship' or 'Rule Waiver'
      """

      #example input: '{d}[\s\S]*?('
      #example output: '{d}[\s\S]*?(Contingency Funding[\s\S]*?(?=(?:Finance Rule|Space Reservation|Sponsorship|$)))'
      
      if (type(start_list) is not list):
         raise Exception('cont_appender_helper start_list argument is not list')
      elif len(start_list) == 0:
         raise Exception('cont_appender_helper start_list argument is empty')
      elif len(start_list) == 1:
          pattern += start_list[0]
      else: 
         for start_keyword in start_list[:-1]: 
            pattern += start_keyword + '|'
         pattern += start_list[-1]
      
      pattern += '[\s\S]*?(?=(?:'

      if (type(end_list) is not list):
         raise Exception('cont_appender_helper end_list argument is not list')
      elif len(end_list) == 0:
         raise Exception('cont_appender_helper end_list argument is empty')
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
      final_pattern = cont_appender_helper(initial_pattern, start, end)
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

def Cont_Approval_Dataframe(dict):
   """
   Converts the nested dictionary produced by `Cont_Approval_Helper` into a Pandas DataFrame.

   This DataFrame organizes the extracted agenda data, with columns for meeting dates, club names, Ficomm decisions, and allocated amounts (if applicable).

   Args:
      dict (dict): A dictionary containing the processed agenda data, where each date maps to a sub-dictionary of clubs and their Ficomm decisions.

   Returns:
      pandas.DataFrame: A DataFrame with columns 'Ficomm Meeting Date', 'Organization Name', 'Ficomm Decision', and 'Amount Allocated'.
   
   Notes:
      - The `Amount Allocated` column will contain the parsed dollar amount if available; otherwise, it will be set to `-1` if the value is not a number.
      - This function is designed to work with the dictionary structure returned by `Cont_Approval_Helper`.
   """
   
   dates_list = []
   club_list = []
   result_list = []
   amt_list = []
   for date in dict:
      for club in dict[date]:
         dates_list.append(date)
         club_list.append(club)
         result_list.append(dict[date][club])
         try: 
            amt_list.append(int(dict[date][club]))
         except Exception as e: 
            amt_list.append(-1)
   rv = pd.DataFrame({'Ficomm Meeting Date' : dates_list, 'Organization Name' : club_list, 'Ficomm Decision' : result_list, 'Amount Allocated' : amt_list})
   return rv

def Cont_Approval(input_txt):
   """
   Processes a raw agenda string and converts it into a Pandas DataFrame with club funding decisions.

   This function combines `Cont_Approval_Helper` for extracting the data and `Cont_Approval_Dataframe` to format it into a DataFrame for easier analysis.

   Args:
      input_txt (str): The raw text of the Ficomm agenda to be processed.

   Returns:
      pandas.DataFrame: A DataFrame containing columns for the meeting date, organization name, Ficomm decision, and the amount allocated (if specified).

   Notes:
      - This function provides an easy interface for extracting and formatting agenda data into a DataFrame.
      - It handles agenda sections related to contingency funding and applies custom start/end keyword filters if provided.
   """
   return Cont_Approval_Dataframe(Cont_Approval_Helper(input_txt))
