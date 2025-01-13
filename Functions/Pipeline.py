import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import re
import unittest
import pytest
import spacy
nlp_model = spacy.load("en_core_web_md")
from sklearn.metrics.pairwise import cosine_similarity 
from rapidfuzz import fuzz, process

from .Utils import Column_Converter

def SU_Cont_Processor(df):
    """
    Expected Intake: Df with following columns: 
    """

def OASIS_Standard_Processor(df):
    """
    Expected Intake: Df with following columns: 
    """

def Ficomm_Dataset_Processor(ficomm_agenda, FR_sheets, OASISMaster):
    """
    Expected Intake: Df with following columns: 
    """