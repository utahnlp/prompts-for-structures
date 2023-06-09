import numpy as np
import pandas as pd
import spacy
from sklearn.metrics import f1_score
from typing import List

from graph import get_all_cliques
from utils import right_to_left_search, Config, get_modified_ans, check_violations



def get_cluster(clusters, ent_id):
    for c_ix, clus in enumerate(clusters):
        if ent_id in clus:
            return c_ix


def create_coref_dumps(rel_rows, gold_clus, pred_clus, g_base_id, p_base_id, gold_dump, pred_dump):
    sent_ids = []
    sentences = []
    gold_ent_tags = []
    pred_ent_tags = []
    mention_traversed = []
   
    with open(gold_dump, "a+") as f:
        f.write(f"""#begin document ({rel_rows[0]['doc_id']});\n""")
    with open(pred_dump, "a+") as f:
        f.write(f"""#begin document ({rel_rows[0]['doc_id']});\n""")


    for row in rel_rows:
        doc_id = row['doc_id']
        if row["sent1_id"] not in sent_ids:
            sent_ids.append(row["sent1_id"])
            sentences.append(row["sent1"])
            gold_ent_tags.append(["_"]*len(row['sent1']))
            pred_ent_tags.append(["_"]*len(row['sent1']))
        if row["sent2_id"] not in sent_ids:
            sent_ids.append(row["sent2_id"])
            sentences.append(row["sent2"])
            gold_ent_tags.append(["_"]*len(row['sent2']))
            pred_ent_tags.append(["_"]*len(row['sent2']))
        
        if row["mention_id1"] not in mention_traversed:
            gold_cluster_id =  get_cluster(gold_clus, row["mention_id1"])
            pred_cluster_id =  get_cluster(pred_clus, row["mention_id1"])
            
            sent_ref_id = sent_ids.index(row["sent1_id"]) # Location of the corresponding sentence
           
            if len(row["ent1_ix"]) == 1:
                gold_ent_tags[sent_ref_id][row["ent1_ix"][0]] = f"({gold_cluster_id+g_base_id})"
                pred_ent_tags[sent_ref_id][row["ent1_ix"][0]] = f"({pred_cluster_id+p_base_id})"
            else:
                gold_ent_tags[sent_ref_id][row["ent1_ix"][0]] = f"({gold_cluster_id+g_base_id}"
                pred_ent_tags[sent_ref_id][row["ent1_ix"][0]] = f"({pred_cluster_id+p_base_id}"
                gold_ent_tags[sent_ref_id][row["ent1_ix"][-1]] = f"{gold_cluster_id+g_base_id})"
                pred_ent_tags[sent_ref_id][row["ent1_ix"][-1]] = f"{pred_cluster_id+p_base_id})"

            mention_traversed.append(row["mention_id1"])

        if row["mention_id2"] not in mention_traversed:
            gold_cluster_id =  get_cluster(gold_clus, row["mention_id2"])
            pred_cluster_id =  get_cluster(pred_clus, row["mention_id2"])

            sent_ref_id = sent_ids.index(row["sent2_id"])

            if len(row["ent2_ix"]) == 1:
                gold_ent_tags[sent_ref_id][row["ent2_ix"][0]] = f"({gold_cluster_id+g_base_id})"
                pred_ent_tags[sent_ref_id][row["ent2_ix"][0]] = f"({pred_cluster_id+p_base_id})"
            else:
                gold_ent_tags[sent_ref_id][row["ent2_ix"][0]] = f"({gold_cluster_id+g_base_id}"
                pred_ent_tags[sent_ref_id][row["ent2_ix"][0]] = f"({pred_cluster_id+p_base_id}"
                gold_ent_tags[sent_ref_id][row["ent2_ix"][-1]] = f"{gold_cluster_id+g_base_id})"
                pred_ent_tags[sent_ref_id][row["ent2_ix"][-1]] = f"{pred_cluster_id+p_base_id})"


            mention_traversed.append(row["mention_id2"])
    
     
    last_words = 0
    for ix, sent in enumerate(sentences):
        sent_id = sent_ids[ix]
        for w_ix, word in enumerate(sent): 
            with open(gold_dump, "a+") as f:
                f.write(f"""{doc_id}\t{sent_id}\t{last_words+w_ix}\t{word}\t{gold_ent_tags[ix][w_ix]}\n""")
            with open(pred_dump, "a+") as f:
                f.write(f"""{doc_id}\t{sent_id}\t{last_words+w_ix}\t{word}\t{pred_ent_tags[ix][w_ix]}\n""")

        last_words += len(sent)

    with open(gold_dump, "a+") as f:
        f.write(f"""\n""")
        f.write("#end document\n")
    with open(pred_dump, "a+") as f:
        f.write(f"""\n""")
        f.write("#end document\n")


    return g_base_id+len(gold_clus), p_base_id+len(pred_clus)

       

def eval_ontonotes(data, preds, meta):
    """ Evaluate Coref dataset
    """
    doc = None
    gold_ans = []
    gold_relation_ids = []
    pred_relation_ids = []
    pred_relation_ids_no = []

    all_relation_ids = []
    post_inf_ans = []
    rel_rows = []
    max_nodes = 0

    gold_base_id = 0
    pred_base_id = 0

    with open(meta['gold_dump_file'], "w+") as f:
        f.write("")
        #f.write("#begin document (Coref);\n")
    with open(meta['pred_dump_file'], "w+") as f:
        f.write("")
        #f.write("#begin document (Coref);\n")

    g_viol = 0
    p_viol = 0
    num_transitivity = 0

    for ix, row in data.iterrows():
        if doc == None:
            doc = row["doc_id"]
        # Change in doc_id implies a new structure
        if doc != row["doc_id"]:
            #if meta["constrained"]:
            #    gold_clus, gold_violations = get_all_cliques(gold_relation_ids, max_nodes)
            #    pred_clus, pred_violations = get_all_cliques(pred_relation_ids, max_nodes)
        
            gold_clus, gold_violations  = right_to_left_search(gold_relation_ids, max_nodes)
            pred_clus, _  = right_to_left_search(pred_relation_ids, max_nodes)
            
            pred_violations, total_checks = check_violations(pred_relation_ids, pred_relation_ids_no, max_nodes)

            g_viol += gold_violations
            p_viol += pred_violations
            num_transitivity += total_checks
            if not meta['constrained']:
                modified_ans = get_modified_ans(pred_clus, all_relation_ids)
                post_inf_ans.extend(modified_ans)

            gold_base_id, pred_base_id = create_coref_dumps(rel_rows, gold_clus, pred_clus, gold_base_id, pred_base_id, meta['gold_dump_file'], meta['pred_dump_file'])

            # Refresh List
            gold_relation_ids = []
            pred_relation_ids = []
            pred_relation_ids_no = []
            all_relation_ids = []
            doc = row["doc_id"]
            max_nodes = 0
            rel_rows = []

        gold_ans.append(row['answer'])  # List which stores the gold answers
        max_nodes = max(max_nodes, row["mention_id1"]+1, row["mention_id2"]+1)
        rel_rows.append(row)
        # Curate edges to form the clusters ultimately

        
        if row['answer'] == 'Yes':
            gold_relation_ids.append([row['mention_id1'], row["mention_id2"]])
        if preds[ix] == 'Yes':
            pred_relation_ids.append([row['mention_id1'], row["mention_id2"]])
        elif preds[ix] == 'No':
            pred_relation_ids_no.append([row['mention_id1'], row["mention_id2"]])

        all_relation_ids.append([row['mention_id1'], row["mention_id2"]])


    # For the last document
    gold_clus, gold_violations = right_to_left_search(gold_relation_ids, max_nodes)
    pred_clus, _ = right_to_left_search(pred_relation_ids, max_nodes)

    pred_violations, total_checks = check_violations(pred_relation_ids, pred_relation_ids_no, max_nodes)

    g_viol += gold_violations
    p_viol += pred_violations
    num_transitivity += total_checks

    if not meta['constrained']:
        modified_ans = get_modified_ans(pred_clus, all_relation_ids)
        post_inf_ans.extend(modified_ans)

    _, _ = create_coref_dumps(rel_rows, gold_clus, pred_clus, gold_base_id, pred_base_id, meta['gold_dump_file'], meta['pred_dump_file'])



    f1 = f1_score(gold_ans, preds, average='macro')
    print(f"F1 Score (Pre-inference): {f1}")
    if not meta['constrained']:
        f1_post = f1_score(gold_ans, post_inf_ans, average='macro')
        print(f"F1 Score (Post-inference): {f1_post}")
    #print(f"Gold Violations: {g_viol}")
    if not meta['constrained']:
        print(f"Transitivity Violations (Prediciton): {p_viol}")
        print(f"Total transitivity checks: {num_transitivity}")


