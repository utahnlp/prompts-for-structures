import pandas as pd


def prompt_srl_wiki(data, config):
    """ Generate prompts and their corresponding answer.
    """
    if config.prompt_type == "discrete":
        prompts = []
        gold = []

        for ix, row in data.iterrows():
            if config.model == "t5":
                prompts.append(f"""question: {row["question"]} context: {row["sentence"]}""")
            gold.append(row["answer"])
        
    return prompts, gold


PROMPT_DICT = {

            "srl": {
                    "wiki" : prompt_srl_wiki
                },
        }

def generate_prompts(data: pd.DataFrame, config):
    """ Generates prompts based on the data and configuration.
    """
    return PROMPT_DICT[config.task_name][config.dataset_name](data, config)
