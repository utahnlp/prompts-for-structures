import itertools
import pandas as pd
from utils import get_highlighted_context


def prompt_srl_wiki(data, config):
    """ Generate prompts and their corresponding answer.
    """
    if config.prompt_type == "discrete":
        prompts = []
        gold = []

        for ix, row in data.iterrows():
            if config.model in ["t5","t5-11b","t5-3b"]:
                prompts.append(f"""question: {row["question"]} context: {row["sentence"]}""")
            elif config.model == "unified-qa":
                prompts.append(f"""{row["question"]} \n {row["sentence"]} """)
            gold.append(row["answer"])
        
    return prompts, gold


def prompt_qasrl2(data, config):
    """ Generate prompts and their corresponding answer.
    """
    if config.prompt_type == "discrete":
        prompts = []
        gold = []
        
        for ix, row in data.iterrows():
            if config.model in ["t5","t5-11b","t5-3b"]:
                prompts.append(f"""question: {row["question"]} context: {" ".join(row["sentence"])} """)
            elif config.model == "unified-qa":
                prompts.append(f"""{row["question"]} \n {" ".join(row["sentence"])} """)

            gold.append(row["answer"])
    
    return prompts, gold



def prompt_coref_ecbplus(data,config):
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
            #if ix == 10:
            #    for p in prompts:
            #        print(p)
    

            if config.model in ["t5","t5-11b","t5-3b"]:
                if config.prompt_style == "nli": 
                    prompts.append(f"""hypothesis: {row["entity1"]} refers to {row["entity2"]}.  premise: {sent} """)
                if config.prompt_style == "qa":
                    prompts.append(f"""question: Does {row["entity1"]} refer to {row["entity2"]}? Yes or No? context: {sent} """)
                elif config.prompt_style == "mcq":
                    prompts.append(f"""copa choice1: Yes choice2: premise: {sent} question: Does {row["entity1"]} refer to {row["entity2"]}? Yes or No?""")
            elif config.model in ["macaw-3b"]:
                prompts.append(f"""$answer$ ; $mcoptions$=(A) Yes (B) No  ; {sent} Does {row["entity1"]} refer to {row["entity2"]}?""")
                
                #first = 1
                #second = 2
                #if not row["in_order"]:
                #    first = 2
                #    second = 1

                #if row["sent1"] != row["sent2"]:
                #    s1 = " ".join(row[f"sent{first}"])
                #    context = f"""wsc: {s1} """
                #else:
                #    context = "wsc: "

                #e2_ix = row[f"ent{second}_ix"]
                #s2 = []
                #for w_ix, w in enumerate(row[f"sent{second}"]):
                #    if w_ix == e2_ix[0]:
                #        if len(e2_ix) == 1:
                #            s2.append("*"+w+"*")
                #        else:
                #            s2.append("*"+w)
                #    elif w_ix == e2_ix[-1]:
                #        s2.append(w+"*")
                #    else:
                #        s2.append(w)

                #context += " ".join(s2)
                #prompts.append(context)
                #gold.append(row[f"entity{first}"])
            gold.append(row["answer"])

    return prompts, gold




PROMPT_DICT = {

            "srl": {
                    "wiki" : prompt_srl_wiki,
                    "qasrl2": prompt_qasrl2
                },
            "coref":{
                    "ecbplus": prompt_coref_ecbplus
                },
        }

def generate_prompts(data: pd.DataFrame, config):
    """ Generates prompts based on the data and configuration.
    """
    return PROMPT_DICT[config.task_name][config.dataset_name](data, config)
