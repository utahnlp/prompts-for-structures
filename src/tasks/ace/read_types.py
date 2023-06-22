import csv
from collections import defaultdict
import pickle
import jsonlines


with open(f"/Users/valentinapy/PycharmProjects/prompts-for-structures/data/dumps_vero_types/dumps/ace_types_ace_types_macaw-3bace_types_vero_gens.bin", "rb") as out:
    gens = pickle.load(out)

infile = csv.reader(open('/Users/valentinapy/PycharmProjects/prompts-for-structures/data/dumps_vero_types/dumps/type_questions_yesno.csv'))

outfile = csv.writer(open('type_preds.tsv', 'w'))

#arg_dict[sent_id+predicate+arg_role]: valid arg + probability
arg_dict = defaultdict(lambda : [])
#sent_id+predicate+arg_role+type : arg+count :
counter = defaultdict(lambda : 0)
gold_dict = defaultdict(lambda : '')
for item, row in zip(gens, infile):
    pred = item[0]['sentence']
    if 'Yes' in pred:
        try:
            count = counter[row[4] + row[5] + row[7] + row[-2]+row[2]]
            preds = row[10].split('%%%')
            scores = row[11].split('%%%')
            pred = preds[count]
            score = float(scores[count])
            arg_dict[row[4]+row[5]+row[7]].append({"sentence": pred, "score": score})
            gold_dict[row[4]+row[5]+row[7]] = row[2]
        except IndexError:
            pass
    counter[row[4] + row[5] + row[7] + row[-2]+row[2]] += 1
    print(counter)
    row.append(pred)
    outfile.writerow(row)

out = []
gold_list = []
with open(f"filtered_gens_vero.bin", "wb") as outfile:
    for key, value in arg_dict.items():
        gold_list.append(gold_dict[key])
        out+=value
    pickle.dump(out, outfile)

with open(f"gold_vero.bin", "wb") as outfile:
    pickle.dump(gold_list, outfile)