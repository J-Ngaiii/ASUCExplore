import sys
print(f"Sys path: {sys.path}")

def get_my_python_path():
    
    import sys
    PATHS = sys.path
    
    num = 1
    print('\nMy PYTHONPATH: Where Python searches when importing modules (lower number takes precedence):')
    print('-'*91)
    
    for path in PATHS:
        print('{}. {}'.format(num, path))
        num += 1
        
get_my_python_path()

try:
    import ASUCExplore.Cleaning as cl
    from ASUCExplore.Cleaning import in_df
    from ASUCExplore import is_type
    print("Cleaning.py works!")
except Exception as e:
    raise e

try:
    import ASUCExplore.Utils as ut
    from ASUCExplore.Utils import heading_finder
    from ASUCExplore import column_converter
    print("Utils.py works!")
except Exception as e:
    raise e

try:
    from ASUCExplore.Core import *
    from ASUCExplore.Core import ABSA_Processor
    print("Core works!")
except Exception as e:
    raise e

try:
    from ASUCExplore.Processor import ASUCProcessor
    from ASUCExplore import ASUCProcessor
    print("Processor.py works!")
except Exception as e:
    raise e
