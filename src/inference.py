from graph import construct_graph, Graph
from gurobipy import Model, GRB, quicksum, abs_

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, List

from tasks.srl.wikisrl.inference import inference_srl
from tasks.coref.ecbp.inference import inference_coref
from tasks.ace.inference import inference_ace

INFERENCE_DICT = {
            "srl" : inference_srl,
            "coref": inference_coref,
            "ace" : inference_ace,
            "ace_type" : inference_ace
        }


def run_inference(task: str, data: pd.DataFrame, generations: List[List[dict]], sanity: bool) -> List[str]:
    """ Nodal method for computing inference. This implies that the input and output
    data types should be fixed.
    Inputs
    -------------------------
    task: str. Task name as specified in the config files.
    data: pd.DataFrame. The preprocessed data. This might be the output of your 
            preprocessing method.
    generations: List[List[str]]. Generations from the language model. Each element contains a list
            of all the paths in the generation beam in descending order of score/probability. Each 
            path contains a its corresponding sentence and score.
    sanity: bool. If True, gold answers are taken to check for perfect inference. 

    Outputs
    ------------------------
    const_ans: List[str]. List of answers corresponding to the prompts after applying the inference.
    """
    return INFERENCE_DICT[task](data, generations, sanity)
