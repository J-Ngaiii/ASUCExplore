import pandas as pd
from typing import Callable
import re
from ASUCExplore.Cleaning import is_type
from ASUCExplore.Core import ABSA_Processor, Agenda_Processor, OASIS_Abridged, FR_ProcessorV2


class ASUCProcessor:
    """Wrapper class for processors. Specify the file type (eg. ABSA) then the __call__ method executes the appropriate processing function, outputting the result.
    The get_type method also outputs the type of processing (eg. ABSA processing pipeline) the ASUCProcessor instance was instructed to execute. 
    Both the actual output of processing (list of processed pd.DataFrame objects) and the type of processing initiated (self.type) are returned to an upload function. 
    
    Processing functions must take in:
    - df_dict (dict[str:pd.DataFrame]): dictionary where keys are file ids and values are the raw files converted into pandas dataframes.
    - reporting (str): parameter that tells the processing function whether or not to print outputs on processing progress.
    - names (dict[str:str]): dictionary where keys are file ids and values are raw file names.
    
    Processing functions must return:
    - list of processed pd.DataFrame objects
    - list of names with those failing naming conventions highighted
        - highlighting means we append 'MISMATCH' to the beginning the name
    
    Higher level architecture:
    - drive_pull func --> outputs raw files as dataframes and list of raw file names
    - ASUCProcessor instance 
        - takes in list of raw files names and raw fils as dataframes

        --> outputs processed fils in a list, type of processing executed and refined list of names with naming convention mismatches flagged
    - drive_push func:
        - From ASUCProcessor instance: take in the outputs of the processed files, the type of processing executed and updated list of names
        
        --> adjust the names of the files accodingly to indicate they're cleaned (based on raw file name and type of processing initiated) then upload files back into ocfo.database drive.

    Dependencies:
    - Currently depends on having ABSA_Processor from ASUCExplore > Core > ABSA_Processor.py alr imported into the file
    """

    def __init__(self, process_type: str):
        self.type = process_type.upper()
        self.processors = {
            'ABSA': self.absa,
            'CONTINGENCY': self.contingency,
            'OASIS': self.oasis, 
            'FR': self.fr
        }
        if self.type not in self.processors:
            raise ValueError(f"Invalid process type '{self.type}'")
        
    process_configs = {
        "ABSA" : {
            'Raw Tag': "RF", 
            'Clean Tag': "GF", 
            'Clean File Name': "ABSA", 
            'Raw Name Dependency': ["Date"], # raw files need to have the date in their file name
            'Processing Function': ABSA_Processor}, 
        "CONTINGENCY" : {
            'Raw Tag': "RF", 
            'Clean Tag': "GF", 
            'Clean File Name': "Ficomm-Cont", 
            'Raw Name Dependency': None, 
            'Processing Function': Agenda_Processor}, 
        "OASIS" : {
            'Raw Tag':"RF", 
            'Clean Tag':"GF", 
            'Clean File Name':"OASIS", 
            'Raw Name Dependency':["Date"], 
            'Processing Function':OASIS_Abridged}, 
        "FR" : {
            'Raw Tag':"RF", 
            'Clean Tag':"GF", 
            'Clean File Name':"Ficomm-", 
            'Raw Name Dependency':["Date", "Numbering", "Coding"], 
            'Processing Function':FR_ProcessorV2}
    }

    # ----------------------------
    # Basic Getter Methods
    # ----------------------------

    def get_type(self):
        return self.type   

    @staticmethod
    def get_process_configs():
        return ASUCProcessor.process_configs
    
    # ----------------------------
    # Config Getter Methods
    # ----------------------------
    
    @staticmethod
    def get_config(process: str, key: str) -> str:
        return ASUCProcessor.get_process_configs().get(process.upper(), {}).get(key)
    
    def get_tagging(self, tag_type = 'Raw') -> str:
        process_dict = ASUCProcessor.get_process_configs()
        match tag_type:
            case 'Raw':
                query = 'Raw Tag'
            case 'Clean':
                query = 'Clean Tag'
            case _:
                raise ValueError(f"Unkown tag type {tag_type}. Please specify either 'Raw' or 'Clean'")
        return process_dict.get(self.get_type()).get(query)
    
    def get_file_naming(self, tag_type = 'Clean') -> str:
        process_dict = ASUCProcessor.get_process_configs()
        match tag_type:
            case 'Clean':
                query = 'Clean File Name'
            case _:
                raise ValueError(f"Unkown tag type {tag_type}")
        return process_dict.get(self.get_type()).get(query)
    
    def get_name_dependency(self) -> str:
        process_dict = ASUCProcessor.get_process_configs()
        return process_dict.get(self.get_type()).get('Raw Name Dependency')
    
    def get_processing_func(self) -> str:
        process_dict = ASUCProcessor.get_process_configs()
        return process_dict.get(self.get_type()).get('Processing Function')
    
    # ----------------------------
    # Validation Methods
    # ----------------------------

    def processor_validations(self, df_dict, names, datatype = pd.DataFrame):
        assert isinstance(df_dict, dict), f"df_dict is not a dictionary but {type(df_dict)}"
        assert is_type(list(df_dict.keys()), str), f"df_dict keys are not all strings, keys: {list(df_dict.keys())}"
        assert is_type(list(df_dict.values()), datatype), f"df_dict values are not {datatype}, values: {list(df_dict.values())}"

        assert isinstance(names, dict), f"names is not a dictionary but {type(names)}"
        assert is_type(list(names.keys()), str), f"names keys are not all strings, keys: {list(df_dict.keys())}"
        assert is_type(list(names.values()), str), f"names values are not strings, values: {list(df_dict.values())}"

        assert len(df_dict) == len(names), f"Given {len(df_dict)} dataframe(s) but {len(names)} name(s)"

        if not df_dict:
            raise ValueError("df_dict is empty! No DataFrames to process.")
        if not names:
            raise ValueError("names is empty! No file names to process.")

        return True
    
    # ----------------------------
    # Processor Methods
    # ----------------------------
    
    def absa(self, df_dict, names, reporting = False) -> list[pd.DataFrame]:
        # need to check if df_dict and names are the same length but handle for case when name is a single string
        assert self.processor_validations(df_dict, names)

        df_lst = list(df_dict.values())
        id_lst = list(df_dict.keys())
        name_lst = list(names.values())

        rv = []
        for i in range(len(df_lst)):
            try: 
                df = df_lst[i]
                id = id_lst[i]
                name = name_lst[i]

                # Name Validation
                mismatch = False
                year_match = re.search(r'(?:FY\d{2}|fr\d{2}|\d{2}\-\d{2}\|\d{4}\-\d{4}\)|\d{2}_\d{2})', name)
                if not year_match:
                    print(f"No valid year in file name\nFile name: {name}\nID: {id}")
                    mismatch = True
                year = year_match[0]

                if self.get_type().lower() not in name.lower():
                    print(f"File does not matching processing naming conventions!\nFile name: {name}\nID: {id}")
                    mismatch = True

                if mismatch:
                    name_lst[i] = 'MISMATCH-' + name_lst[i] # WARNING: mutating array as we loop thru it, be careful
                else:
                    validated_name = f"{self.get_file_naming(tag_type = 'Clean')}-{year}-{self.get_tagging(tag_type = 'Raw')}" # ABSA draws from ficomm files formatted "ABSA-date-RF"
                    name_lst[i] = validated_name
                
                # Processing
                try:
                    processing_function = self.get_processing_func()
                    rv.append(processing_function(df))
                except Exception as e:
                    print(f"Processing function {self.get_processing_func().__name__} errored while processing file {name}, ID {id}\nPassing error from {self.get_processing_func().__name__}")
                    raise e
                if reporting:
                    print(f"Successfully ran {self.get_processing_func().__name__} on File: {name}, id: {id}")
            except Exception as e:
                raise e
        return rv, name_lst
    
    def contingency(self, txt_dict, names, reporting = False) -> list[pd.DataFrame]:
        """
        Function that takes in a dictionary of txt files and names then outputs a dictionary of processed txt files with updated names. 
        Date is appended to updated file names under formatting: %m/%d/%Y.
        """
        assert self.processor_validations(txt_dict, names, datatype=str)
        
        txt_lst = list(txt_dict.values())
        id_lst = list(txt_dict.keys())
        name_lst = list(names.values())

        rv = []
        for i in range(len(txt_lst)):
            try: 
                txt = txt_lst[i]
                id = id_lst[i]
                name = name_lst[i]
                
                processing_function = self.get_processing_func()
                try:
                    output, date = processing_function(txt,  debug=False)
                except Exception as e:
                    print(f"Processing function {self.get_processing_func().__name__} errored while processing file {name}, ID {id}\nPassing error from {self.get_processing_func().__name__}")
                    raise e
                date_formatted = pd.Timestamp(date).strftime("%m/%d/%Y")
                rv.append(output)

                validated_name = f"{self.get_file_naming(tag_type = 'Clean')}-{date_formatted}-{self.get_tagging(tag_type = 'Raw')}" # Contingency draws from ficomm files formatted "Ficomm-date-RF"
                name_lst[i] = validated_name
                if 'ficomm' not in name.lower() and 'finance committee' not in name.lower():
                    print(f"File does not matching processing naming conventions!\nFile name: {name}\nID: {id}") # do we raise to stop program or just print?
                    name_lst[i] = 'MISMATCH-' + name_lst[i] # WARNING: mutating array as we loop thru it, be careful
                if reporting:
                    print(f"Successfully ran {self.get_processing_func().__name__} on File: {name}, id: {id}")
            except Exception as e:
                raise e
        return rv, name_lst
    
    def oasis(self, df_dict, names, reporting = False) -> list[pd.DataFrame]:
        assert self.processor_validations(df_dict, names)
        
        df_lst = list(df_dict.values())
        id_lst = list(df_dict.keys())
        name_lst = list(names.values())

        rv = []
        for i in range(len(df_lst)):
            try: 
                df = df_lst[i]
                id = id_lst[i]
                name = name_lst[i]

                # Name Validation
                mismatch = False
                year_match = re.search(r'(?:FY\d{2}|fr\d{2}|\d{2}\-\d{2}\|\d{4}\-\d{4}\)|\d{2}_\d{2})', name)
                if not year_match:
                    print(f"No valid year in file name\nFile name: {name}\nID: {id}")
                    mismatch = True
                year = year_match[0]

                if self.get_type().lower() not in name.lower():
                    print(f"File does not matching processing naming conventions!\nFile name: {name}\nID: {id}")
                    mismatch = True

                if mismatch:
                    name_lst[i] = 'MISMATCH-' + name_lst[i] # WARNING: mutating array as we loop thru it, be careful
                else:
                    validated_name = f"{self.get_file_naming(tag_type = 'Clean')}-{year}-{self.get_tagging(tag_type = 'Raw')}" # ABSA draws from ficomm files formatted "ABSA-date-RF"
                    name_lst[i] = validated_name
                    
                # Processing
                try:
                    processing_function = self.get_processing_func()
                    rv.append(processing_function(df, year))
                except Exception as e:
                    print(f"Processing function {self.get_processing_func().__name__} errored while processing file {name}, ID {id}\nPassing error from {self.get_processing_func().__name__}")
                    raise e
                if reporting:
                    print(f"Successfully ran {self.get_processing_func().__name__} on File: {name}, id: {id}")
            except Exception as e:
                raise e
        return rv, name_lst
    
    def fr(self, df_dict, names, reporting = False) -> list[pd.DataFrame]:
        assert self.processor_validations(df_dict, names)
        
        df_lst = list(df_dict.values())
        id_lst = list(df_dict.keys())
        name_lst = list(names.values())

        rv = []
        for i in range(len(df_lst)):
            try: 
                df = df_lst[i]
                id = id_lst[i]
                name = name_lst[i]

                # Name Validation
                mismatch = False
                year_match = re.search(r'(?:FY\d{2}|fr\d{2}|\d{2}\-\d{2}\|\d{4}\-\d{4}\)|\d{2}_\d{2})', name)
                if not year_match:
                    print(f"No valid year in file name\nFile name: {name}\nID: {id}")
                    mismatch = True
                year = year_match[0]

                numbering_match = re.search(r'(?:F|S)\d{2}', name)
                if not numbering_match:
                    print(f"No valid numbering detected in name detected for file {name} (ID: {id})")
                    mismatch = True
                number = numbering_match[0]

                if self.get_type().lower() not in name.lower():
                    print(f"File does not matching processing naming conventions!\nFile name: {name}\nID: {id}")
                    mismatch = True

                if mismatch:
                    name_lst[i] = 'MISMATCH-' + name_lst[i]
                else:
                    validated_name = f"{self.get_file_naming(tag_type = 'Clean')}-{year}-{number}-{self.get_tagging(tag_type = 'Raw')}" # ABSA draws from ficomm files formatted "ABSA-date-RF"
                    name_lst[i] = validated_name

                # Processing
                try:
                    processing_function = self.get_processing_func()
                    rv.append(processing_function(df, debug=True))
                except Exception as e:
                    print(f"Processing function {self.get_processing_func().__name__} errored while processing file {name}, ID {id}\nPassing error from {self.get_processing_func().__name__}")
                    raise e
                if reporting:
                    print(f"Successfully ran {self.get_processing_func().__name__} on File: {name}, id: {id}")
            except Exception as e:
                raise e
        return rv, name_lst
        
    # A little inspo from CS189 HW6
    def __call__(self, df_dict: dict[str, pd.DataFrame], names: dict[str, str], reporting: bool = False) -> list[pd.DataFrame]:
        """Call the appropriate processing function based on type."""
        if self.type not in self.processors:
            raise ValueError(f"Unsupported processing type '{self.type}'")
        return self.processors[self.type](df_dict, names, reporting) 