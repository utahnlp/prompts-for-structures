
import pickle



with open(f"/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/dumps_argument_types/dumps/ace_ace_t5_gens.bin", "rb") as out:
    gens_types = pickle.load(out)
with open(f"/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/dumps_argument_types/dumps/ace_ace_t5_gold", "rb") as out:
    gold_types = pickle.load(out)


with open(f"/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/dumps_questions_args/dumps/ace_ace_t5_gens.bin", "rb") as out:
    gens = pickle.load(out)
with open(f"/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/dumps_questions_args/dumps/ace_ace_t5_gold", "rb") as out:
    gold = pickle.load(out)

for g in gens_types:
    print(g)

#for g in gold_types:
#    print(g)