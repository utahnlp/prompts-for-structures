import csv
from collections import defaultdict
import pickle
import jsonlines


with open(f"/Users/valentinapy/PycharmProjects/prompts-for-structures/data/dumps_types_yesno2/dumps/ace_types_ace_types_macaw-3bace_types_gens.bin", "rb") as out:
    gens = pickle.load(out)

infile = csv.reader(open('/Users/valentinapy/PycharmProjects/prompts-for-structures/data/dumps_types_yesno2/type_questions_yesno.csv'))

outfile = csv.writer(open('type_preds.tsv', 'w'))

#arg_dict[sent_id+predicate+arg_role]
arg_dict = defaultdict(lambda : [])
for item, row in zip(gens, infile):
    print(item)
    pred = item[0]['sentence']
    row.append(pred)
    outfile.writerow(row)

out = []

with open(f"filtered_gens.bin", "wb") as outfile:
    pickle.dump(out, outfile)