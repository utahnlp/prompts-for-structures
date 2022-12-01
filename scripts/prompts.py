import pandas as pd


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
            if config.model in ["t5","t5-11b","t5-3b"]:
               # prompts.append(f"""hypothesis: {row["entity1"]} refers to {row["entity2"]}.  premise: {row["sentence"]} """)
               prompts.append(f"""question: Does {row["entity1"]} refer to {row["entity2"]}? Yes or No? context: {row["sentence"]} """)
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
