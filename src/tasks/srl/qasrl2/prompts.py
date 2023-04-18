import pandas as pd
from typing import Tuple, List


def prompt_qasrl2(data: pd.DataFrame, config) -> Tuple[List[str], List[str]]:
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
            if config.model in ["t5","t5-11b","t5-3b","t5-small","t5-base"]:
                prompts.append(f"""question: {row["question"]} context: {" ".join(row["sentence"])} """)
            elif config.model == "unified-qa":
                prompts.append(f"""{row["question"]} \\n {" ".join(row["sentence"])} """)
            elif config.model == "flan-t5-xl":
                prompts.append(f"""{" ".join(row["sentence"])} \\n In the above sentence, {row["question"]}""")
            gold.append(row["answer"])
    
    return prompts, gold


