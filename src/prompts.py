import itertools
import pandas as pd
from typing import Tuple, List
from utils import get_highlighted_context, Config

from tasks.srl.wikisrl.prompts import prompt_srl_wiki
from tasks.srl.qasrl2.prompts import prompt_qasrl2
from tasks.coref.ecbp.prompts import prompt_coref_ecbplus


PROMPT_DICT = {

            "srl": {
                    "wiki" : prompt_srl_wiki,
                    "qasrl2": prompt_qasrl2
                },
            "coref":{
                    "ecbplus": prompt_coref_ecbplus,
                    "ontonotes": prompt_coref_ecbplus,
                    "genia": prompt_coref_ecbplus
                },
        }




def generate_prompts(data: pd.DataFrame, config: Config) -> Tuple[List[str], List[str]]:
    """ Generates prompts based on the data and configuration. This is the nodal
    function for all the prompt generation process. This means that the inputs
    and outputs for this function are fixed in data type across all tasks and 
    datasets.
    Inputs
    -----------------------
    data: pd.DataFrame. This contains all the data to generate prompts from. 
            This typically is the output of the preprocess function which means
            that each row in the dataframe essentially contains the content of 
            one prompt.
    config: utils.Config. Config class instance which contains the meta-data.
            The meta data also contains prompt types, styles, and context styles.

    Outputs
    -----------------------
    l1: List[str]. List of prompts to query the language model. 
        Example element for coref: "question: e1 and e2 refer to the same entity? Yes or No? context: <context>"
    l2: List[str]. List of gold answers to the respective prompts in l1.
        Example element for coref: "Yes"
    """
    return PROMPT_DICT[config.task_name][config.dataset_name](data, config)
