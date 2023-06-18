import csv
from collections import defaultdict
import pickle
import jsonlines


with open(f"/Users/valentinapy/PycharmProjects/prompts-for-structures/data/dumps_types_yesno/dumps/ace_ace_t5-3b_gens.bin", "rb") as out:
    gens = pickle.load(out)

for item in gens:
    print(item)
