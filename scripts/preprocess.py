import pandas as pd
from pathlib import Path
from typing import Union


def preprocess_wikisrl(filepath):
    """ Preprocessing function for Wikipedia SRL.
    """
    with open(filepath) as f:
        data = f.readlines()
    
    sent_id = None
    total_predicates = None
    predicate = None
    sentence = None

    processed_data = []
    
    for line in data:
        if line!="\n":
            split_data = line.strip("\n").split("\t")
            if len(split_data) == 2:
                sent_id = split_data[0]
                total_predicates = int(split_data[1])
            elif len(split_data) == 1:
                sentence = line.strip("\n")
            elif len(split_data) == 3:
                predicate = split_data[1]
            else:
                answer = split_data[-1]
                # Iterating over all question tokens
                # skipping dashes for a coherent question
                ques_str = ""
                for ques_tok in split_data[:-1]:
                    if ques_tok != "_":
                        ques_str += f"{ques_tok} "
                ques_str = ques_str.strip() # Remove the trailing space

                processed_data.append([sent_id, total_predicates, sentence, predicate, ques_str, answer])
    
    columns = ["sent_id","total_predicates","sentence","predicate","question","answer"]
    data_df = pd.DataFrame(processed_data, columns = columns)

    return data_df




PREPROCESS_DICT = {
            "srl" : {
                    "wiki": preprocess_wikisrl
                },

        }



def preprocess_file(file_path: Union[str,Path], task: str, dataset: str):
    try:
        return PREPROCESS_DICT[task][dataset](file_path)
    except KeyError:
        print("****Please check your task_name and dataset_name in your config file. They should match the dictionary keys in \
                PREPROCESS_DICT in preprocess.py****")
