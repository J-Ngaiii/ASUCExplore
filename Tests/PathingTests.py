import sys

import sys
print(f"Sys path: {sys.path}")

from src.Special import ABSA_Processor
import src.Cleaning as cl
import src.Utils as ut

print(f"Utils Func: {ut.heading_finder}")
print(f"Cleaning Func: {cl.get_valid_iter}")
print(f"ABSA Processor: {ABSA_Processor}")

print("All print statements completed, pathing functional!")
