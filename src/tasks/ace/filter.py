import csv
from collections import defaultdict
import pickle
import jsonlines
#
# id_file = csv.reader(open('/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/type_questions.csv'))
#
# id_arg_file = csv.reader(open('/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/argument_questions.csv'))
#
#
# with open(f"/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/dumps_argument_types/dumps/ace_ace_t5_gens.bin", "rb") as out:
#     gens_types = pickle.load(out)
# with open(f"/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/dumps_argument_types/dumps/ace_ace_t5_gold", "rb") as out:
#     gold_types = pickle.load(out)
#
#
# with open(f"/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/dumps_questions_args/dumps/ace_ace_t5_gens.bin", "rb") as out:
#     gens = pickle.load(out)
# with open(f"/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/dumps_questions_args/dumps/ace_ace_t5_gold", "rb") as out:
#     gold = pickle.load(out)
#
#
# # id:pred:arg : [preds]
# type_dict = defaultdict(lambda : [])
# for id_row, row in zip(id_file, gens_types):
#     type_dict[id_row[1]+id_row[2]+id_row[4]].append(row)
#
# # id:pred:arg : [preds]
# arg_dict = defaultdict(lambda : [])
# for id_row, row in zip(id_arg_file, gens):
#     arg_dict[id_row[1]+id_row[2]+id_row[4]] = row
#
# out = []
# for key, value in arg_dict.items():
#     inner_out = []
#     types = type_dict[key]
#     covered = {}
#     for type_list in types:
#         for t in type_list:
#             covered[t['sentence']] = 'yes'
#     for sent in value:
#         sentence = sent['sentence']
#         if sentence in covered:
#             print('yes')
#             inner_out.append(sent)
#         else:
#             print('no')
#     out.append(inner_out)
# with open(f"filtered_gens.bin", "wb") as outfile:
#     pickle.dump(out, outfile)

def filter_input_file():
    infile = jsonlines.open('/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/ace_dev.jsonl')
    outfile = jsonlines.open('/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/ace_dev_filtered.jsonl', mode='w')
    covered = {}
    for row in infile:
        sent_id = row['predicate']['event_id']
        predicate = row['predicate']['lemma']
        out = row
        arg_list = []
        for arg in row['arguments']:
            arg_role = arg['role_type']
            if sent_id+predicate+arg_role not in covered:
                arg_list.append(arg)
                covered[sent_id+predicate+arg_role] = True
        out['arguments'] = arg_list
        outfile.write(out)

filter_input_file()