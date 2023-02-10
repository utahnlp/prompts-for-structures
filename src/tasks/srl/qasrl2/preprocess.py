import json
import jsonlines
import pandas as pd
from pathlib import Path
from typing import Union


def preprocess_qasrl2(filepath: Union[str, Path]) -> pd.DataFrame:
    """ Preprocessing function for QA SRL 2
    Input
    ----------------------
    filepath: str or pathlib.Path. Input data file path

    Output
    ----------------------
    data_df: pd.DataFrame. Dataframe where each row represents a row
            for prompting.
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
                         
                    processed_data.append([sent_id, total_predicates, sentence, predicate, ques_str, ans_str, ans_spans])
    
    columns = ["sent_id","total_predicates","sentence","predicate","question","answer","ans_span"]
    data_df = pd.DataFrame(processed_data, columns = columns)

    return data_df


