import csv

import pickle

id_file = csv.reader(open('/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/type_questions.csv'))

with open(f"/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/dumps_argument_types/dumps/ace_ace_t5_gens.bin", "rb") as out:
    gens_types = pickle.load(out)
with open(f"/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/dumps_argument_types/dumps/ace_ace_t5_gold", "rb") as out:
    gold_types = pickle.load(out)


with open(f"/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/dumps_questions_args/dumps/ace_ace_t5_gens.bin", "rb") as out:
    gens = pickle.load(out)
with open(f"/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/dumps_questions_args/dumps/ace_ace_t5_gold", "rb") as out:
    gold = pickle.load(out)


for id_row, row in zip(id_file, gens_types):
