import json
import jsonlines
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




def preprocess_qasrl2(filepath):
    """ Preprocessing function for QA SRL 2
    """
    with jsonlines.open(filepath) as f:
        sent_id = None
        total_predicates = None
        predicate = None
        sentence = None
        
        processed_data = []
    
        for line in f.iter():
            sent_id = line["sentenceId"]
            total_predicates = len(line["verbEntries"])
            sentence = line["sentenceTokens"]
            
            for verb_key, verbval in line["verbEntries"].items():
                predicate = verbval['verbInflectedForms']['stem']
                for ques, ans_det in verbval["questionLabels"].items():
                    ques_str = ans_det["questionString"]
                    ans_spans = []
                    ans_str = ""
                    for ans in ans_det["answerJudgments"]:
                        if not ans["isValid"]:
                            continue
                        for ans_sp in ans["spans"]:
                            new_ans_str = sentence[ans_sp[0]:ans_sp[1]]
                            if ans_sp not in ans_spans:
                                ans_spans.append(ans_sp)
                                if ans_str == "":
                                    ans_str += " ".join(new_ans_str)
                                else:
                                    ans_str += f""" ### {" ".join(new_ans_str)}"""
                        #if len(ans["spans"]) != 1:
                        #    print(sentence)
                        #    print(predicate)
                        #    print(ques_str)
                        #    print(ans_str)
                        #    exit())
                    
                    processed_data.append([sent_id, total_predicates, sentence, predicate, ques_str, ans_str, ans_spans])
    
    columns = ["sent_id","total_predicates","sentence","predicate","question","answer","ans_span"]
    data_df = pd.DataFrame(processed_data, columns = columns)

    return data_df



            






PREPROCESS_DICT = {
            "srl" : {
                    "wiki": preprocess_wikisrl,
                    "qasrl2": preprocess_qasrl2
                },

        }



def preprocess_file(file_path: Union[str,Path], task: str, dataset: str):
    #try:
    return PREPROCESS_DICT[task][dataset](file_path)
   # except KeyError:
     #   print("****Please check your task_name and dataset_name in your config file. They should match the dictionary keys in \
     #           PREPROCESS_DICT in preprocess.py****")
