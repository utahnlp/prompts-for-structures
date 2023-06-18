import pandas as pd
from typing import Tuple, List
import itertools
import pandas as pd
from typing import Tuple, List
from utils import get_highlighted_context, Config
import re

def prompt_ace(data: pd.DataFrame, config) -> Tuple[List[str], List[str]]:
    """ Generate prompts and their corresponding answer.
    Inputs
    -----------------
    data: pd.DataFrame. Input dataframe containing preprocessed data
            Each row should contain the ingredients for constructing
            the prompt. It should contain at least a "question" key
            which contains the SRL question, "sentence" key which
            contains the context and the "answer" key which contains
            the gold answer.

    Outputs
    -----------------
    prompts: List[str]. A list of all prompts for the language models.
    gold   : List[str]. A list of all gold answers for all the corresponding prompt.
    """
    if config.prompt_type == "discrete":
        prompts = []
        gold = []

        for ix, row in data.iterrows():
            if config.model in ["t5", "t5-11b", "t5-3b"]:
                prompts.append(f"""question: {row["question"]} context: {row["sentence"]} """)
            elif config.model == "unified-qa":
                prompts.append(f"""{row["question"]} \n {row["sentence"]} """)
            gold.append(row["answer"])

    return prompts, gold


def prompt_ace_types_yes_no(data: pd.DataFrame, config: Config) -> Tuple[List[str], List[str]]:
    """ Creating Prompts from the processed data according to the input config.
    Inputs
    --------------------
    data: pd.DataFrame. Processed data which might be the output of yout preprocessing method
    config: Config. The config data from the input config file

    Outputs
    -------------------
    prompts - List[str]. The list of prompts which will be fed to the model.
    gold    - List[str]. A parallel list to prompts which contains the gold answers.
    """
    if config.prompt_type == "discrete":
        prompts = []
        gold = []

        for ix, row in data.iterrows():
            sent = row["sentence"]
            question = row["question"]
            question = re.sub('this', row['answer'], question)
            if config.model in ["t5", "t5-11b", "t5-3b"]:
                prompts.append(f"""question: {question} context: {row["sentence"]} """)
            elif config.model in ["macaw-3b"]:
                prompts.append(
                    f"""$answer$ ; $mcoptions$=(A) Yes (B) No  ; {sent} {question}""")

            gold.append(row["answer"])

    return prompts, gold