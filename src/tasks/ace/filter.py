import csv
from collections import defaultdict
import pickle
import jsonlines


def read_type_questions():
    #type : question
    type_q_map = defaultdict(lambda : '')
    infile = csv.DictReader(open('../data/questions/questions.tsv'), delimiter='\t')

    for row in infile:
        type_q_map[row['type']]=row['question']

    return type_q_map

def read_types():
    # predicate : argument : types
    type_dict = defaultdict(lambda: defaultdict(lambda: []))
    infile = csv.reader(open('../data/questions/type_question_map.tsv'), delimiter='\t')

    for row in infile:
        if not 'Time' in row[0]:
            predicate, arg = row[0].split('_')
            predicate = '_'.join(predicate.split('.'))
            type_dict[predicate][arg] = row[2:]

    return type_dict

def read_questions():
    # predicate : argument : question
    q_dict = defaultdict(lambda : defaultdict(lambda : ''))
    infile = csv.reader(open('/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/questions/type_question_map.tsv'), delimiter='\t')

    for row in infile:
        if not 'Time' in row[0]:
            predicate, arg = row[0].split('_')
            predicate = '_'.join(predicate.split('.'))
            q_dict[predicate][arg] = row[1]

    return q_dict

def read_gold_attributes():
    attrib_dict = defaultdict(lambda : '')
    infile = csv.DictReader(open('/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/attributes.csv'))
    for row in infile:
        attrib_dict[row['id']]=row['type']
    return attrib_dict

id_file = csv.reader(open('/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/type_questions.csv'))

id_arg_file = csv.reader(open('/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/argument_questions.csv'))


# with open(f"/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/dumps_argument_types/dumps/ace_ace_t5_gens.bin", "rb") as out:
#     gens_types = pickle.load(out)
# with open(f"/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/dumps_argument_types/dumps/ace_ace_t5_gold", "rb") as out:
#     gold_types = pickle.load(out)


with open(f"/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/dumps_questions_args2/dumps/ace_ace_t5_gens.bin", "rb") as out:
    gens = pickle.load(out)
with open(f"/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/dumps_questions_args2/dumps/ace_ace_t5_gold", "rb") as out:
    gold = pickle.load(out)

outfile = csv.DictWriter(open('readable_argument_preds_improved.tsv', 'w'), fieldnames=['sentence', 'predicate', 'event_type', 'event_id', 'gold_argument', 'gold_span', 'role_type', 'arg_type', 'predicted_arguments', 'arg_question'])
outfile.writeheader()

# id:pred:arg : [preds]
type_dict = defaultdict(lambda : [])
for id_row, row in zip(id_file, gens_types):
    type_dict[id_row[1]+id_row[2]+id_row[4]].append(row)

# id:pred:arg : [preds]
arg_dict = defaultdict(lambda : [])
for id_row, row in zip(id_arg_file, gens):
    arg_dict[id_row[1]+id_row[2]+id_row[4]] = row


infile = jsonlines.open('/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/ace_dev.jsonl')
attrib_dict = read_gold_attributes()
q_dict = read_questions()
for row in infile:
    sent_id = row['predicate']['event_id']
    sentence = row['text']
    predicate = row['predicate']['lemma']
    predicate_role = row['predicate']['event_type']
    # TODO: for now I am only predicting for existing arguments!
    for arg in row['arguments']:
        arg_role = arg['role_type']
        if 'Time' in arg_role:
            pass
        else:
            if predicate_role in q_dict:
                if arg_role in q_dict[predicate_role]:
                    question = q_dict[predicate_role][arg_role]
                else:
                    print(predicate_role)
                    print(arg_role)
                    outfile.writerow([predicate_role, arg_role])
            else:
                print(predicate_role)
                print(arg_role)
            ques_str = question
            ans_str = arg["text"]
            ans_span = arg["span"]
            ans_span = ans_span.split(':')
            ans_span = [[int(ans_span[0]), int(ans_span[1])]]
            arg_type = attrib_dict[arg['arg_id']]
            predictions = arg_dict[sent_id+predicate+arg_role]
            predictions = [pred['sentence'] for pred in predictions]
            print(predicate)
            print(predicate_role)
            print(sent_id)
            print(ans_str)
            print(ans_span)
            print(arg_role)
            print(arg_type)
            print(predictions)
            print(question)
            outfile.writerow({'sentence':sentence, 'predicate':predicate, 'event_type':predicate_role, 'event_id':sent_id, 'gold_argument':ans_str, 'gold_span': ans_span, 'role_type':arg_role, 'arg_type':arg_type, 'predicted_arguments':'%%%'.join(predictions), 'arg_question':question})

# out = []
# for key, value in arg_dict.items():
#     inner_out = []
#     types = type_dict[key]
#     covered = {}
#     for type_list in types:
#         for t in type_list:
#             covered[t['sentence']] = 'yes'
#     first_sent = value[0]
#     for sent in value:
#         sentence = sent['sentence']
#         if sentence in covered:
#             print('yes')
#             inner_out.append(sent)
#         else:
#             print('no')
#     #TODO: think about this
#     if len(inner_out)==0:
#         inner_out.append(first_sent)
#     out.append(inner_out)
# with open(f"filtered_gens.bin", "wb") as outfile:
#     pickle.dump(out, outfile)

def filter_input_file():
    infile = jsonlines.open('/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/ace_dev.jsonl')
    outfile = jsonlines.open('/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/ace_dev_filtered_by_pred_arg.jsonl', mode='w')
    covered = {}
    for row in infile:
        sent_id = row['predicate']['event_id']
        predicate = row['predicate']['span']
        out = row
        arg_list = []
        for arg in row['arguments']:
            arg_role = arg['span']
            if sent_id+predicate+arg_role not in covered:
                arg_list.append(arg)
                covered[sent_id+predicate+arg_role] = True
        if len(arg_list)>0:
            out['arguments'] = arg_list
            outfile.write(out)

