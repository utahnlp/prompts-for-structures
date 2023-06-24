import csv
from collections import defaultdict
import pickle
import jsonlines


infile = csv.reader(open('/Users/valentinapy/PycharmProjects/prompts-for-structures/data/readable_argument_preds_veronica_nulls_for_types.tsv'))
gold_list = []
out = []
for row in infile:
    gold = row[2]
    gold_list.append(gold)
    args = row[10].split('%%%')
    scores = row[11].split('%%%')
    if float(scores[0]) < -0.6:
        out.append([{"sentence": 'None', "score": float(scores[0])}])
    else:
        inner_out = []
        for arg, score in zip(args, scores):
            score = float(score)
            inner_out.append({"sentence": arg, "score": score})
        out.append(inner_out)

with open(f"../../../data/filtered_gens_vero_null.bin", "wb") as outfile:
    pickle.dump(out, outfile)
with open(f"../../../data/gold_vero_null.bin", "wb") as outfile:
    pickle.dump(gold_list, outfile)