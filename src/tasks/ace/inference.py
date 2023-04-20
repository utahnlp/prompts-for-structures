from graph import construct_graph, Graph
from gurobipy import Model, GRB, quicksum, abs_

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List


def inference_ace(data: pd.DataFrame, generations, sanity_check) -> List[str]:
    """ Constrained inference for the SRL task
    Inputs
    ------------
    data - pd.DataFrame. Processed dataset
    generations- List[List[{"sentence":str, "score":float}]]. List of responses generated by the
                prompt model. Size: (|dataset|,|beam|,2). The dictionary contains a candidate response
                along with its score.
    sanity_check - bool. If True, the gold answers are moved to top of the beam.  This helps to test the
                inference algorithm in the ideal case.

    Outputs
    ------------
    const_ans: List[str]. List of answers after constraining output
    """
    predicate = None
    sentence = None
    sent_id = None
    pred_gens = []
    const_ans = []
    gold_ans = []
    gold_ans_spans = []
    invalid_gold = 0

    cnt_ix = -1
    # iterate over the dataset
    for ix, row in tqdm(data.iterrows()):
        print(row)
        cnt_ix += 1
        if predicate == None:
            predicate = row["predicate_lemma"]
            sentence = row["text"]
            #sent_id = row["sent_id"]

        # If thecondition is satisfied, we have the data points for the structure
        #if ((predicate != row["predicate_lemma"]) or (sent_id != row["sent_id"])):
        if ((predicate != row["predicate_lemma"])):
            predicate = row["predicate_lemma"]
            #sent_id = row["sent_id"]

            c_ans, g_inv = construct_graph(sentence, pred_gens, ix, gold_ans, sanity_check, ans_span=gold_ans_spans)
            const_ans += c_ans  # Answers selected via the inference algorithm
            invalid_gold += g_inv  # All answers that are invalid (non-extarctive)

            sentence = row["text"]
            pred_gens = []
            gold_ans = []
            gold_ans_spans = []

        # Store answers and the gold answers
        pred_gens.append(generations[cnt_ix])
        gold_ans.append(row["argument_text"])
        # Gold answer spans for sanity check
        if "argument_span" in row.keys():
            gold_ans_spans.append(row["argument_span"])

    # Inference for the last structure
    c_ans, g_inv = construct_graph(sentence, pred_gens, len(data), gold_ans, sanity_check, ans_span=gold_ans_spans)
    const_ans += c_ans
    invalid_gold += g_inv
    print(f"# Gold answers not perfect sub-sequences: {invalid_gold}")

    return const_ans


