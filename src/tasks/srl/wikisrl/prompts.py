from itertools import permutations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

from tasks.srl.wikisrl.preprocess import preprocess_wikisrl


def build_prompt(config, row):
    if config.model in ["t5","t5-11b","t5-3b","t5-base","t5-small"]:
        return f"""question: {row["question"]} context: {row["sentence"]}"""
    elif (config.model == "unified-qa") or (config.model[:12]=="unifiedqa-v2"):
        return f"""{row["question"].lower()} \n {row["sentence"].lower()} """
    elif config.model == "flan-t5-xl":
        return f"""{" ".join(row["sentence"])} \n In the above sentence, {row["question"]}"""
    else:
        print("""************* Model name not supported for the dataset-task combination. Please recheck model name. """)




def get_few_shot(few_shot_df, config):
    """ Get the few-shot example strings. 
    Inputs
    --------------
    few_shot_df: pd.DataFrame. Dataframe with few shot examples
    config: utils.Config. The input config file.
    """
    if config.shot_type == "same":
        shots = few_shot_df.sample(n=config.num_shots, random_state = config.shot_seed)
    else:
        shots = few_shot_df.sample(n=config.num_shots)

    shot_prompts = []
    for ix, row in shots.iterrows():
        ques_context = build_prompt(config, row)
        shot_prompts.append(ques_context + f""" {row["answer"]}""")
    
    nums = list(range(config.num_shots))
    perms = list(map(list,permutations(nums,config.num_shots)))

    prompt_order = perms[config.order_num-1]
    shot_text = ""
    for ix in prompt_order:
        if config.model[:2] == "t5":
            shot_text += shot_prompts[ix] + " "
        else:
            shot_text += shot_prompts[ix] + "\n"

    return shot_text


def prompt_srl_wiki(data: pd.DataFrame, config) -> Tuple[List[str], List[str]]:
    """ Generate prompts and their corresponding answer.
    Inputs 
    -----------------
    data: pd.DataFrame. Input dataframe containing preprocessed data
            Each row should contain the ingredients for constructing 
            the prompt. It should contain at least a "question" key 
            which contains the SRL question, "sentence" key which 
            contains the context and the "answer" key which contains 
            the gold answer.
    config: utils.Config. The config dictionary which is the main input
            to the system

    Outputs
    -----------------
    prompts: List[str]. A list of all prompts for the language models.
    gold   : List[str]. A list of all gold answers for all the corresponding prompt.
    """
    if config.few_shot:
        few_shot_df = preprocess_wikisrl(Path(config.data_dir,config.few_shot_file))
        np.random.seed(42)
    if config.prompt_type == "discrete":
        prompts = []
        gold = []

        for ix, row in data.iterrows():
            shot_text = ''
            if config.few_shot:
                shot_text = get_few_shot(few_shot_df, config)
            prompts.append(shot_text + build_prompt(config, row))# + " answer: ")
            gold.append(row["answer"])
        
    return prompts, gold


