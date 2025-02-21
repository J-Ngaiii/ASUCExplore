# ASUCExplore Package
This package is a collection of the data cleaning and pipeline functions I developed when cleaning and analyzing ASUC Contingency data. They're designed to operate on reports and CSVs generated out of Student Union Finance and OASIS at UC Berkeley. More information can be seen in the 'ASUC-Finance-Budget-Analysis' repo. 

Much of the functions are built with Pandas to balance computationally efficient vectorized operations with the functionalities of Pandas. 

# Status of Functions and Modules
## ABSA Processor Module: Fully Tested and Functional
ABSA Processor Function: Tested and Functional 

## Cleaning: Partially Tested and Functional
- is_type: Tested and Functional   
- in_df: Tested and Functional  
- any_in_df: Not Formally Tested
- concatonater: Not Formally Tested
- academic_year_parser: Tested but Not Functional
- reverse_academic_year_parser: Not Tested
- type_test: Tested and Functional
- row_test: Tested and Functional
- col_test: Tested and Functional
- col_mismatch_test: Tested and Functional
- cat_migration_checker: Tested and Functional

## Utils: Partially Tested and Functional 
- column_converter: Tested and Functional
- column_renamer: Tested and Functional
- any_drop: Not Formally Tested
- oasis_cleaner: Tested and Functional
- sucont_cleaner: Tested and Functional
- bulk_manual_populater: Tested and Functional
- category_updater: Tested and Functional
- heading_finder: Not Formally Tested
