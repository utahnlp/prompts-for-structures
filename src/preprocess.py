from conllu import parse, parse_incr
import glob
import json
import jsonlines
import os
import pandas as pd
from pathlib import Path
from typing import Union
from itertools import combinations
from utils import dataset_document_iterator

from tasks.srl.wikisrl.preprocess import  preprocess_wikisrl
from tasks.srl.qasrl2.preprocess import preprocess_qasrl2
from tasks.coref.ecbp.preprocess import preprocess_ecbplus_coref
from tasks.ace.preprocess import preprocess_ace


def preprocess_ontonotes_coref(filepath):
    onto_fname = f"./../dumps/onto_agg.txt"
    # Clear file
    with open(onto_fname, "w+") as f:
        pass
    
    onto_file = open(onto_fname, "a+")
    onto_file.write("#begin document\n")

    for f in glob.iglob(str(filepath)+"/**/**.gold_conll", recursive=True):
        docs = dataset_document_iterator(f)
        for doc_ix, doc in enumerate(docs):
            word_id = 1
            for sent_ix, sent in enumerate(doc):
                open_mention = False
                open_ent = None
                proc = []
                for word_ix, word in enumerate(sent):
                    info = word.split()
                    
                    count_op = info[-1].count('(')
                    count_cl = info[-1].count(')')
                    # We need to remove nested entities
                    if info[-1] == "-":
                        lab = info[-1]
                    elif (info[-1][0] == "(") and (count_op >= count_cl):
                        if open_mention:
                            lab = "-"
                        else:
                            if '|' not in info[-1]:
                                lab = info[-1]
                                if info[-1][-1] != ")":
                                    open_mention = True
                                    open_ent = info[-1].strip('(')
                            else:
                                # Take the last open entity which corresponds
                                # to the longest entity
                                rel_ents = info[-1].split("|")
                                for ent_ix in range(len(rel_ents)-1, -1, -1):
                                    if rel_ents[ent_ix][-1] != ")":
                                        lab = rel_ents[ent_ix]
                                        open_mention = True
                                        open_ent = rel_ents[ent_ix].strip("(")
                                        break

                            
                    elif info[-1][-1] == ")":
                        if '|' not in info[-1]:
                            if info[-1].strip(')') == open_ent:
                                lab = info[-1].split('|')[-1]
                                open_mention = False
                                open_ent = None
                            else:
                                lab = '-'
                        else:
                            rel_ents = info[-1].split("|")
                            for ent_ix in range(len(rel_ents)-1, -1, -1):
                                if rel_ents[ent_ix][0] != "(":
                                    if rel_ents[ent_ix].strip(')') == open_ent:
                                        lab = rel_ents[ent_ix]
                                        open_mention = False
                                        open_ent = None
                                        break
                                lab = '-'




                    onto_file.write(f"{info[0]}\t{info[1]}\t{info[0]}_{info[1]}\t{sent_ix}\t{word_id}\t{info[3]}\tTrue\t{lab}\n")
                    word_id += 1

                try:
                    assert not open_mention
                except AssertionError:
                    print(f"WARNING: Unclosed mentions")
                    print(f)
                    for lab in proc:
                        print(lab)
                    exit()
            
    onto_file.write(f"#end document")
    onto_file.close()

    data_df = preprocess_ecbplus_coref(onto_fname)
    return data_df
        
    
    




PREPROCESS_DICT = {
            "srl" : {
                    "wiki": preprocess_wikisrl,
                    "qasrl2": preprocess_qasrl2
                },
            "coref" : { 
                    "ecbplus": preprocess_ecbplus_coref,
                    "ontonotes": preprocess_ontonotes_coref
                },
            "ace" : {"ace": preprocess_ace}
        }



def preprocess_file(file_path: Union[str,Path], task: str, dataset: str) -> pd.DataFrame:
    """ This is the nodal method for all preprocessing function for 
    all tasks. This implies that the input and output data types should 
    be fixed. 
    Inputs
    ---------------------
    file_path: str or pathlib.Path. The file path to the input data.
    task: str. Task name as set in the config file.
    dataset: str. Data set name as set in the config file.

    Outputs
    ---------------------
    df: pd.DataFrame. The dataframe which contains the processed data. Ideally,
        each row pertains to one component which would be prompted to the 
        language model. For example, for the task of coreference, a row will
        denote a mention-pair for which we want a "Coref or Not Coref (Yes/No)" 
        answer from the model.
    """
    #try:
    return PREPROCESS_DICT[task][dataset](file_path)
   # except KeyError:
     #   print("****Please check your task_name and dataset_name in your config file. They should match the dictionary keys in \
     #           PREPROCESS_DICT in preprocess.py****")
