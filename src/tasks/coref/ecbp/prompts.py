import itertools
import pandas as pd
from typing import Tuple, List
from utils import get_highlighted_context, Config



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
    if config.prompt_type == "discrete":
        prompts = []
        gold = []
        
        for ix, row in data.iterrows():
            sent = row["sentence"]  
            if config.context_style == "highlight":
                sent = get_highlighted_context(row, config.model)
            elif config.context_style == "full_context":
                sent = " ".join(itertools.chain(*row['passage']))
            elif config.context_style == "highlight_full_context":
                sent = get_highlighted_context(row, config.model, full_context=True)
    

            if config.model in ["t5","t5-11b","t5-3b"]:
                if config.prompt_style == "nli": 
                    prompts.append(f"""hypothesis: {row["entity1"]} refers to {row["entity2"]}.  premise: {sent} """)
                if config.prompt_style == "qa":
                    prompts.append(f"""question: Does {row["entity1"]} refer to {row["entity2"]}? Yes or No? context: {sent} """)
                elif config.prompt_style == "mcq":
                    prompts.append(f"""copa choice1: Yes choice2: premise: {sent} question: Does {row["entity1"]} refer to {row["entity2"]}? Yes or No?""")
            elif config.model in ["macaw-3b"]:
                # The if condition below is to counter OOM errors for GENIA
                if config.dataset_name == "genia":
                    sent = " ".join(sent.split()[:90]) 
                prompts.append(f"""$answer$ ; $mcoptions$=(A) Yes (B) No  ; {sent} Does {row["entity1"]} refer to {row["entity2"]}?""")
                
            gold.append(row["answer"])

    return prompts, gold


