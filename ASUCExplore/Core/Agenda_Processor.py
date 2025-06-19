import numpy as np
import pandas as pd
import re

from ASUCExplore.Cleaning import is_type

def _find_chunk_pattern(starts, ends, end_prepattern = '\d\.\s') -> str:
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

def _motion_processor(club_names, names_and_motions) -> dict[str:list[str]]:
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

def Agenda_Processor(inpt, start=['Contingency Funding', 'Contingency'], end=['Finance Rule', 'Rule Waiver', 'Space Reservation', 'Sponsorship', 'Adjournment', 'ABSA', 'ABSA Appeals'], identifier='(\w+\s\d{1,2}\w*,\s\d{4})') -> pd.DataFrame:
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

   valid_name_chars = '\w\s\-\_\*\&\%\$\+\#\@\!\(\)\,\'\"\.' #seems to perform better with explicit handling for special characters? eg. for 'Telegraph+' we add the plus sign so regex will pick it up
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
