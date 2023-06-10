import itertools
import pandas as pd
from typing import Tuple, List
from utils import get_highlighted_context, Config

from tasks.coref.ecbp.preprocess import preprocess_ecbplus_coref 
from tasks.coref.ontonotes.preprocess import preprocess_ontonotes_coref
from tasks.coref.genia.preprocess import preprocess_genia_coref


def build_prompt(config, row):
    sent = row["sentence"]  
    if config.context_style == "highlight":
        sent = get_highlighted_context(row, config.model)
    elif config.context_style == "full_context":
        sent = " ".join(itertools.chain(*row['passage']))
    elif config.context_style == "highlight_full_context":
        sent = get_highlighted_context(row, config.model, full_context=True)
    
    if config.model in ["t5","t5-11b","t5-3b"]:
        if config.prompt_style == "nli": 
            return f"""hypothesis: {row["entity1"]} refers to {row["entity2"]}.  premise: {sent} """
        if config.prompt_style == "qa":
            return f"""question: Does {row["entity1"]} refer to {row["entity2"]}? Yes or No? context: {sent} """
        elif config.prompt_style == "mcq":
            return f"""copa choice1: Yes choice2: premise: {sent} question: Does {row["entity1"]} refer to {row["entity2"]}? Yes or No?"""
    elif config.model in ["macaw-3b","macaw-large"]:
        # The if condition below is to counter OOM errors for GENIA
        #if config.dataset_name == "genia":
        #    sent = " ".join(sent.split()[:90]) 
        return f"""$answer$ ; $mcoptions$=(A) Yes (B) No  ; {sent} Does {row["entity1"]} refer to {row["entity2"]}?"""
            


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
            shot_text += shot_prompts[ix] + "\n"
        else:
            shot_text += shot_prompts[ix] + "\n"

    return shot_text





def prompt_coref_ecbplus(data: pd.DataFrame, config: Config) -> Tuple[List[str],List[str]]:
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
    if config.few_shot:
        if config.dataset_name == "ecbp":
            few_shot_df = preprocess_ecbplus_coref(Path(config.data_dir,config.few_shot_file))
        elif config.dataset_name == "ontonotes":
            few_shot_df = preprocess_ontonotes_coref(Path(config.data_dir,config.few_shot_file))
        elif config.dataset_name == "genia":
            few_shot_df = preprocess_genia_coref(Path(config.data_dir,config.few_shot_file))
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


