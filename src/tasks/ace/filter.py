
import pickle



with open(f"/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/dumps/ace_ace_t5_gens.bin", "rb") as out:
    gens = pickle.load(out)
with open(f"/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/dumps/ace_ace_t5_gold", "rb") as out:
    gold = pickle.load(out)

for g in gens:
    print(g)