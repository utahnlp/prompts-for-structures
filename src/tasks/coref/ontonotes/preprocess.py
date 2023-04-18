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



def generate_examples(c_dict, doc_id):
    """ Generate positive and negative examples from a document
    """
    window = 2
    examples = []
    entity_list = list(c_dict["entities"].keys())
    traversed = []
    # Iterating over entities
    for ent_id in c_dict["entities"].keys():
        traversed.append(ent_id)
        # We can generate positive examples from entities which
        # occur more than once
        if len(c_dict["entities"][ent_id]) > 1:
            # Curating pairs of mentions for the positive examples
            pairs = list(combinations(c_dict["entities"][ent_id],2))
            for pair in pairs:
                # Extarcting entities
                s_id1 = pair[0]["sent_id"]
                s_id2 = pair[1]["sent_id"]
                
                if abs(s_id1 - s_id2) >= window:
                    continue
                norm_ent1_ids = [i-c_dict["sent_ixs"][s_id1] for i in pair[0]["tok_idx"] ]
                norm_ent2_ids = [i-c_dict["sent_ixs"][s_id2] for i in pair[1]["tok_idx"] ]
                
                ent1 = " ".join([c_dict["sentences"][s_id1][i] for i in norm_ent1_ids])
                ent2 = " ".join([c_dict["sentences"][s_id2][i] for i in norm_ent2_ids])
            
                sent1 = " ".join(c_dict["sentences"][s_id1])
                sent2 = " ".join(c_dict["sentences"][s_id2])

                mention_id1 = pair[0]["mention_id"]
                mention_id2 = pair[1]["mention_id"]

                in_order = True

                if pair[0]["sent_id"] == pair[1]["sent_id"]:
                    context  = f"{sent1}"
                elif pair[0]["sent_id"] < pair[1]["sent_id"]:
                    context = f"{sent1} {sent2}"
                else:
                    in_order = False
                    context = f"{sent2} {sent1}"
                
                examples.append([doc_id, c_dict["sentences"],context, "Yes", ent1, ent2, entity_list.index(ent_id), entity_list.index(ent_id), c_dict["sentences"][s_id1], c_dict["sentences"][s_id2],s_id1,s_id2,norm_ent1_ids, norm_ent2_ids, in_order, ent_id, ent_id, mention_id1, mention_id2])
        
        ## Generate negative examples 
        # Iterate over the entities
        for ent in c_dict["entities"][ent_id]:
            s_id1 = ent["sent_id"]
            norm_ent1_ids = [i-c_dict["sent_ixs"][s_id1] for i in ent["tok_idx"] ]
            ent1 = " ".join([c_dict["sentences"][s_id1][i] for i in norm_ent1_ids])
            sent1 = " ".join(c_dict["sentences"][s_id1])
            mention_id1 = ent["mention_id"]
            # Iterating over entites dissimilar to the one considered
            for neg_ent_id in c_dict["entities"].keys():
                if (neg_ent_id == ent_id) or (neg_ent_id in traversed):
                    continue
                # Iterating over all negative instances of a negative entity
                for neg_ent in c_dict["entities"][neg_ent_id]:
                    s_id2 = neg_ent["sent_id"]
                    if abs(s_id2 - s_id1) >= window:
                        continue
                    norm_ent2_ids = [i-c_dict["sent_ixs"][s_id2] for i in neg_ent["tok_idx"] ]
                    ent2 = " ".join([c_dict["sentences"][s_id2][i] for i in norm_ent2_ids])
                    sent2 = " ".join(c_dict["sentences"][s_id2])
                    
                    ent1_id = ent_id
                    ent2_id = neg_ent_id
                    in_order = True
                    mention_id2 = neg_ent["mention_id"]

                    if s_id1 == s_id2:
                        context  = f"{sent1}"
                        if mention_id1 > mention_id2:
                            in_order = False
                    elif s_id1 < s_id2:
                        context = f"{sent1} {sent2}"
                    else:
                        context = f"{sent2} {sent1}"
                        in_order = False
                    

                    examples.append([doc_id, c_dict["sentences"], context, "No", ent1, ent2, entity_list.index(ent1_id), entity_list.index(ent2_id), c_dict["sentences"][s_id1], c_dict["sentences"][s_id2],s_id1,s_id2,norm_ent1_ids, norm_ent2_ids, in_order, ent1_id, ent2_id,mention_id1, mention_id2])
                    

    return examples



def preprocess_document(filepath: Union[str, Path]) -> pd.DataFrame:
    """ Preprocess the ECB+ corpora Coreference Resolution.
    Inputs
    ----------------------------
    filepath: str or Path. File containing the input data.

    Outputs
    ----------------------------
    data_df: pd.DataFrame. Dataframe containing the processed data. Each 
                row pertains to one component which would be prompted to the 
                language model.    
    """
    # Read the input file
    with open(filepath) as f:
        data = f.readlines()
    
    data = data[1:-1] 
    
    # All the tracking parameters
    doc = []
    document_id = None
    sent_id = None
    sent_cnt_ix = -1
    token_list = None
    targ_cont_flag = False
    mention_id = 0
    coref_dict = {"sentences": [], "entities":{}, "sent_ixs":[]}

    for line in data:
        line_split = line.strip().split("\t")
        if document_id == None:
            document_id = line_split[2]

        # Check if there is a change in document ID.
        # If this happens you have reached the end of the prevous
        # document and need to preprocess thatparticular 
        # document block
        if (line_split[2] != document_id):
            coref_dict["sentences"].append(token_list) # Adding the current sentence to the sentences dictionary
            examples = generate_examples(coref_dict, document_id)   # Process the document
            doc.extend(examples)
            # Refresh the tracking parameters
            sent_id = None
            token_list = None
            sent_cnt_ix = -1
            coref_dict = {"sentences": [], "entities":{}, "sent_ixs": []}
            document_id = line_split[2]
            mention_id = 0
        
        ############################
        # Update token list from each 
        # sentences
        if sent_id != line_split[3]:
            sent_cnt_ix += 1        # Increase sentence count
            coref_dict["sent_ixs"].append(int(line_split[4])-1) # Token where the sentence starts in the document
            if token_list != None:
                coref_dict["sentences"].append(token_list)
                token_list = None 
            sent_id = line_split[3]     # Update sentence ID to the new one

        # Append tokens to the token list 
        if token_list == None:
            token_list = [] 
        token_list.append(line_split[5])
        
        ############################
        # Update all targets
        if line_split[-1] != "-":
            # Case where the target is a single token
            if (line_split[-1][0] == "(") and (line_split[-1][-1] == ")"):
                tok_idx = [int(line_split[4])-1]
                # Save the entity information. Every entity will have a list
                # of all of its mentions which, for each mention, 
                # includes information like the sentence ID where it occurs, 
                #its token ID and its corresponding unique mention  ID
                if line_split[-1][1:-1] not in coref_dict["entities"].keys():
                    coref_dict["entities"][line_split[-1][1:-1]] = []
                coref_dict["entities"][line_split[-1][1:-1]].append({"sent_id": sent_cnt_ix, "tok_idx":tok_idx, "mention_id":mention_id})
                mention_id += 1     #Update mention ID
            # Case where a multi-token mention begins
            elif line_split[-1][0] == "(":
                tok_idx = [int(line_split[4])-1]
                targ_cont_flag = True       # A flag to show that there is a mutli-token mention getting processed
            # Case where a multi-token mention ends
            elif line_split[-1][-1] == ")":
                tok_idx.append(int(line_split[4])-1)
                # Save the entity information. Every entity will have a list
                # of all of its mentions which, for each mention, 
                # includes information like the sentence ID where it occurs, 
                #its token ID and its corresponding unique mention  ID
                if line_split[-1][:-1] not in coref_dict["entities"].keys():
                    coref_dict["entities"][line_split[-1][:-1]] = []
                coref_dict["entities"][line_split[-1][:-1]].append({"sent_id": sent_cnt_ix, "tok_idx":tok_idx, "mention_id":mention_id}) 
                mention_id += 1
                # Resetting the flag and token ID
                tok_idx = None
                targ_cont_flag = False
        else:
            # If a multi-token mention is being processed all the 
            # intermediate token IDs should be added to that mention
            if targ_cont_flag:
                tok_idx.append(int(line_split[4])-1)
                
    # Converting the data into a DataFrame
    columns = ["doc_id","passage","sentence","answer","entity1","entity2","entity1_id","entity2_id","sent1","sent2","sent1_id","sent2_id","ent1_ix","ent2_ix","in_order","ent1_ix_glob", "ent2_ix_glob", "mention_id1", "mention_id2"]
    data_df = pd.DataFrame(doc, columns=columns)
    
    return data_df





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

    data_df = preprocess_document(onto_fname)
    return data_df
 
