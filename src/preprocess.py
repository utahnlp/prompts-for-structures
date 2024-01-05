from conllu import parse, parse_incr
import glob
import json
import jsonlines
import os
import pandas as pd
from pathlib import Path
from typing import Union
from itertools import combinations
from utils import dataset_document_iterator

from tasks.srl.wikisrl.preprocess import  preprocess_wikisrl
from tasks.srl.qasrl2.preprocess import preprocess_qasrl2
from tasks.coref.ecbp.preprocess import preprocess_ecbplus_coref
from tasks.coref.ontonotes.preprocess import preprocess_ontonotes_coref
from tasks.coref.genia.preprocess import preprocess_genia_coref


PREPROCESS_DICT = {
            "srl" : {
                    "wiki": preprocess_wikisrl,
                    "qasrl2": preprocess_qasrl2
                },
            "coref" : { 
                    "ecbplus": preprocess_ecbplus_coref,
                    "ontonotes": preprocess_ontonotes_coref,
                    "genia": preprocess_genia_coref
                },

        }



def preprocess_file(file_path: Union[str,Path], task: str, dataset: str) -> pd.DataFrame:
    """ This is the nodal method for all preprocessing function for 
    all tasks. This implies that the input and output data types should 
    be fixed. 
    Inputs
    ---------------------
    file_path: str or pathlib.Path. The file path to the input data.
    task: str. Task name as set in the config file.
    dataset: str. Data set name as set in the config file.

    Outputs
    ---------------------
    df: pd.DataFrame. The dataframe which contains the processed data. Ideally,
        each row pertains to one component which would be prompted to the 
        language model. For example, for the task of coreference, a row will
        denote a mention-pair for which we want a "Coref or Not Coref (Yes/No)" 
        answer from the model.
    """
    #try:
    return PREPROCESS_DICT[task][dataset](file_path)
   # except KeyError:
     #   print("****Please check your task_name and dataset_name in your config file. They should match the dictionary keys in \
     #           PREPROCESS_DICT in preprocess.py****")
