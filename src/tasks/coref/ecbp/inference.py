from gurobipy import Model, GRB, quicksum, abs_
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, List


def solve_coref(relations, preds, thresh=0.5):
    thresh = 0.5
    max_ent = np.max(relations)+1
    score_mat = np.full((max_ent,max_ent),-np.inf)
    for ix in range(len(preds)):
        for p in preds[ix]:
            if p['sentence'] == "Yes":
                score_y = p['score']

        score_mat[relations[ix][0],relations[ix][1]] = score_y-thresh


    model = Model('Optim')

    y = [[0]*max_ent for i in range(max_ent)]
    for i in range(max_ent):
        for j in range(i+1,max_ent):
            y_str = f"y{i}{j}"
            y[i][j] = model.addVar(vtype=GRB.BINARY, name=y_str)
    
    for i in range(max_ent):
        for j in range(i+1,max_ent):
            for k in range(j+1,max_ent):
                model.addConstr(y[i][j] + y[j][k] - y[i][k] <= 1)
                model.addConstr(y[i][k] + y[i][j] - y[j][k] <= 1)
                model.addConstr(y[i][k] + y[j][k] - y[i][j] <= 1)


    sum_const = 0
    for i in range(max_ent):
        for j in range(i+1, max_ent):
            sum_const += y[i][j]
    expr = abs_(sum_const)
    #model.addConstr(abs_(quicksum(y[i][j] for i in range(max_ent) for j in range(i+1,max_ent))) >= 1e-8)
    model.addConstr(sum_const >= 1)
    obj = quicksum(y[i][j]*score_mat[i][j] for i in range(max_ent) for j in range(i+1,max_ent))
    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()
    
    final_y = [[0]*max_ent for i in range(max_ent)]
    for i in range(max_ent):
        for j in range(i+1,max_ent):
            final_y[i][j] = y[i][j].X
    
    answers = []
    for rel in relations:
        if final_y[rel[0]][rel[1]] == 1:
            answers.append("Yes")
        else:
            answers.append("No")

    #print(final_y)
    return answers





def inference_coref(data, generations, sanity_check):
    """ Constrained inference for the SRL task
    Inputs
    ------------
    data - pd.DataFrame. Processed dataset
    generations- List[List[{"sentence":str, "score":float}]]. List of responses generated by the 
                prompt model. Size: (|dataset|,|beam|,2). The dictionary contains a candidate response
                along with its score.
    sanity_check - bool. If True, the gold answers are moved to top of the beam.  This helps to test the 
                inference algorithm in the ideal case. 
    """
    structure_ix = None
    doc_id = None
    pred_gens = []
    const_ans = []
    gold_ans = []
    relation_ids = []

    for ix, row in tqdm(data.iterrows()):
        if structure_ix == None:
            structure_ix = ix
            doc_id = row['doc_id']
        #if ix ==3:
        #    print(row.keys())
        #    print(row)
        #    exit()
        # When the doc_id changes, we need to consider the data we have
        # for constrained inference
        if doc_id != row['doc_id']:
            struct_ans = solve_coref(relation_ids, pred_gens)
            const_ans.extend(struct_ans)
             
            pred_gens = []
            gold_ans = []
            relation_ids = []
            doc_id = row['doc_id']

        pred_gens.append(generations[ix])
        gold_ans.append(row["answer"])
        if row['in_order']:
            relation_ids.append([row['mention_id1'],row["mention_id2"]])
        else:
            relation_ids.append([row["mention_id2"],row["mention_id1"]])
    
    struct_ans = solve_coref(relation_ids, pred_gens)
    const_ans.extend(struct_ans)

        
    return const_ans

