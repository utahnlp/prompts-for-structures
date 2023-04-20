import numpy as np
import pandas as pd
import spacy
from sklearn.metrics import f1_score
from typing import List

from graph import get_all_cliques
from utils import right_to_left_search, Config

from tasks.srl.wikisrl.evaluate import eval_wikisrl 
from tasks.coref.ecbp.evaluate import eval_ecbplus
from tasks.ace.evaluate import eval_ace



EVALUATION_DICT = {
            "srl": {
                    "wiki": eval_wikisrl,
                    "qasrl2": eval_wikisrl
                },
            "coref": {
                    "ecbplus": eval_ecbplus,
                },
            "ace": {
                    "ace": eval_ace,
            }
        
        }


def evaluate(data: pd.DataFrame, config: Config, preds: List[str], meta: dict):
    """ Nodal method for evaluations for all tasks. This implies that the input 
    and the output data types should be fixed.
    Inputs
    ----------------------
    data: pd.DataFrame. The processed data. This might be the output of your preprocess
            method.
    config: Config. The config class instance.
    preds: List[str]. List of predictions for each prompt to evaluate for.
    meta: dict. A miscellaneous dictionary to accommodate for any task-specific 
            attributes.
    """
    try:
        return EVALUATION_DICT[config.task_name][config.dataset_name](data, preds, meta)
    except KeyError:
        print("*** Please check your task_name and dataset_name in your config file. They should match the dictionary keys in \
                EVALUATION_DICT in evaluate.py***")


