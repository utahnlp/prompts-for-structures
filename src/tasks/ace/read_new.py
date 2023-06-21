import csv
from collections import defaultdict
import pickle
import jsonlines


id_arg_file = csv.reader(open('/Users/valentinapy/PycharmProjects/prompts-for-structures/data/dumps_vero/dumps/argument_questions.csv'))


with open(f"/Users/valentinapy/PycharmProjects/prompts-for-structures/data/dumps_vero/dumps/ace_ace_t5-3bacevero_gens.bin", "rb") as out:
    gens = pickle.load(out)

outfile = csv.writer(open('readable_argument_preds_veronica_for_types.tsv', 'w'))



for id_row, row in zip(id_arg_file, gens):
    predictions = [pred['sentence'] for pred in row]
    scores = [str(pred['score']) for pred in row]
    id_row.append('%%%'.join(predictions))
    id_row.append('%%%'.join(scores))
    outfile.writerow(id_row)