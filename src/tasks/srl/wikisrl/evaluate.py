import numpy as np
import spacy
from sklearn.metrics import f1_score
from graph import subseq_match


nlp = spacy.load("en_core_web_sm")

def get_srl_stats(pred_golds, pred_gens, corr_qs, corr_qs_dep):
    roots_gold = []
    roots_pred = []

    for ix, gold_ent in enumerate(pred_golds):
        possb_ans = gold_ent.split(" ### ")
        r_ans = []
        for opt in possb_ans:
            doc1 = nlp(opt)
            for token in doc1:
                if token.dep_ == "ROOT":
                    r_ans.append(token.text)
        roots_gold.append(r_ans)
        
        if pred_gens[ix] == '':
            roots_pred.append('')
            continue
        doc2 = nlp(pred_gens[ix])
        for token in doc2:
            if token.dep_ == "ROOT":
                roots_pred.append(token.text)
    # comp_corr is a flag to check if 
    # all arguments for a predicate were 
    # correctly extracted or not. 1 implies 
    # complete correctness and 0 otherwise
    comp_corr = 1
    comp_corr_dep = 1
    
    for ix, gold_ent in enumerate(pred_golds):
        exact_match = False
        for g_ent in  gold_ent.split(" ### "):
            if g_ent == pred_gens[ix]:
                corr_qs += 1
                exact_match = True
                break
        if not exact_match:
            comp_corr = 0
        
        try:
            root_match = False
            for r_gold in roots_gold[ix]:
                if r_gold == roots_pred[ix]:
                    corr_qs_dep += 1
                    root_match =True
                    break
            if not root_match:
                comp_corr_dep = 0
        except IndexError:
            print(pred_golds)
            print(pred_gens)
            print(roots_gold)
            print(roots_pred)
            #exit()
    
    #print(roots_gold)
    #print(roots_pred)
    return comp_corr, corr_qs, comp_corr_dep, corr_qs_dep




def get_consistency_violations(sentence, generations):
    indices = []
    viol = 0
    tot = 0
    for gen in generations:
        if gen == "Has a none response .":
            continue
        _ , s_ix, e_ix = subseq_match(gen.split(), sentence)
        gen_indices = []
        for ix in range(len(s_ix)):
            gen_indices.append([s_ix[ix], e_ix[ix]])
        indices.append(gen_indices.copy())

    for i in range(len(indices)-1):
        for j in range(i+1,len(indices)):
            flag = True # Start with always violation
            query_ids = indices[i]
            test_ids = indices[j]
            for quer in query_ids:
                s_ix = quer[0] #Start ID
                e_ix = quer[1]
                for test_q in test_ids:
                    if ((s_ix >= test_q[0]) and (s_ix< test_q[1])) or ((e_ix > test_q[0]) and (e_ix <= test_q[1])):
                        pass    # This implies violation. We need to check for all combinations
                    else:
                        flag = False
                        break
                if not flag:
                    break
            if flag:
                viol += 1
            tot += 1
    #print(sentence)
    #print(generations)
    #print(viol,tot)
    #if viol >0:
    #    exit()
    return viol, tot


def eval_wikisrl(data, preds, meta=None):
    """ Evaluation module for SRL with wikipedia data.
    """
    predicate = None
    sent_id = None
    pred_gens = []
    pred_golds = []
    comp = 0    # Counter for correct predicates
    total_pred = 0  # Total predicates
    corr_qs = 0     # Counter for correct question-answer pair
    total_qs = 0    # Total qurstions
    comp_dep = 0
    corr_qs_dep = 0

    inconsistencies = 0
    tot_checks = 0

    for ix, row in data.iterrows():
        if predicate == None:
            predicate = row['predicate']
            sent_id = row["sent_id"]
            sentence = row["sentence"]

        if (predicate != row['predicate']) or (sent_id != row["sent_id"]):
            # Compute results at every predicate
            predicate = row['predicate']  
            sent_id = row['sent_id']
            
            comp_corr, corr_qs, comp_corr_dep, corr_qs_dep = get_srl_stats(pred_golds, pred_gens, corr_qs, corr_qs_dep)
            
            if type(sentence) != list:
                sent = sentence.split()
            else:
                sent = sentence

            viol, checks = get_consistency_violations(sent, pred_gens)

            total_qs += len(pred_gens)
            comp += comp_corr
            comp_dep += comp_corr_dep
            total_pred += 1
            inconsistencies += viol
            tot_checks += checks

            pred_gens = []
            pred_golds = []
            sentence = row["sentence"]
        

        #if ix == 20:
        #    break
            
        pred_gens.append(preds[ix])
        pred_golds.append(row['answer'])
    
    # same block as the one in the loop
    # This just accounts for the last predicate
    comp_corr, corr_qs, comp_corr_dep, corr_qs_dep = get_srl_stats(pred_golds, pred_gens, corr_qs, corr_qs_dep)
    viol, checks = get_consistency_violations(sent, pred_gens)
    inconsistencies += viol
    tot_checks += checks



    total_qs += len(pred_gens)
    comp += comp_corr
    comp_dep += comp_corr_dep
    total_pred += 1

    print(f"Completely Correct Predicates: {comp/total_pred}")
    print(f"Exact Accuracy for Argument Extraction: {corr_qs/total_qs}")
    print(f"Completely Correct Predicates by Head Accuracy: {comp_dep/total_pred}")
    print(f"Head Accuracy: {corr_qs_dep/total_qs}")
    print(f"Total inconsistencies: {inconsistencies}")
    print(f"Total checks: {tot_checks}")
    print()
 
