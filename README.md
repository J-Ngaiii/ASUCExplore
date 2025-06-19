# ASUCExplore Package
This package is a collection of the data cleaning and pipeline functions I developed when cleaning and analyzing ASUC Contingency data. They're designed to operate on reports and CSVs generated out of Student Union Finance and OASIS at UC Berkeley. The 'ASUC-Finance-Budget-Analysis' repo has a long notebook that explains the idea + initial implementation of many of these functions. The 'ocfo-etl' implements many of the functions detailed here. 

Numpy is used but the primary dependency is Pandas. 

# Co-Dependencies Within Package
The core work loop is this: Script pulls files from database --> Processor.py executes corresponding processing --> Script pushes files back into the database. 

# Status of Functions and Modules
## Terminology
- Field Tested: I used the function on a couple real documents and it worked there
- Unittested: I made unittests and the function passed all relevant unittests
- Deprecated: Function is no longer supported in current version

## Core Files
- ABSA Processor Function: Field Tested and Functional
- Agenda Processor Function: Field Tested and Functional

## Cleaning: Partially Tested and Functional

## Cleaning: Fully Tested and Functional
- is_type: Unittested and Functional   
- in_df: Unittested and Functional  
- any_in_df: Unittested and Functional  
- concatonater: Deprecated
- academic_year_parser: Deprecated
- reverse_academic_year_parser: Deprecated
- type_test: Deprecated
- row_test: Deprecated
- col_test: Deprecated
- col_mismatch_test: Deprecated
- cat_migration_checker: Deprecated

## Utils: Partially Tested and Functional 
- column_converter: Unittested and Functional
- column_renamer: Field Tested and Functional
- oasis_cleaner: Not Tested and Functional
- sucont_cleaner: Not Tested and Functional
- bulk_manual_populater: deprecated
- category_updater: deprecated
- heading_finder: Unittested and Functional
