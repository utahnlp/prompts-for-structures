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
    infile = csv.reader(open('../data/questions/type_question_map2.tsv'), delimiter='\t')

    for row in infile:
        predicate, arg = row[0].split('_')
        predicate = '_'.join(predicate.split('.'))
        type_dict[predicate][arg] = row[2:]

    return type_dict

def read_questions():
    # predicate : argument : question
    q_dict = defaultdict(lambda : defaultdict(lambda : ''))
    infile = csv.reader(open('/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/questions/type_question_map2.tsv'), delimiter='\t')

    for row in infile:
        predicate, arg = row[0].split('_')
        predicate = '_'.join(predicate.split('.'))
        q_dict[predicate][arg] = row[1]

    return q_dict



id_arg_file = csv.reader(open('/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/dumps_questions_args4/argument_questions.csv'))


with open(f"/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/dumps_questions_args4/ace_ace_t5-3b_gens.bin", "rb") as out:
    gens = pickle.load(out)
with open(f"/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/dumps_questions_args4/ace_ace_t5-3b_gold", "rb") as out:
    gold = pickle.load(out)

outfile = csv.DictWriter(open('readable_argument_preds_improved.tsv', 'w'), fieldnames=['sentence', 'predicate', 'event_type', 'event_id', 'gold_argument', 'gold_span', 'role_type', 'arg_type', 'predicted_arguments', 'arg_question'])
outfile.writeheader()


# id:pred:arg : [preds]
arg_dict = defaultdict(lambda : [])
qs = []
for id_row, row in zip(id_arg_file, gens):
    arg_dict[id_row[1]+id_row[2]+id_row[4]] = row
    qs.append(id_row[0])

infile = jsonlines.open('/Users/valentinapyatkin/PycharmProjects/prompts-for-structures/data/ace_dev_filtered_by_pred_arg.jsonl')
q_dict = read_questions()
counter = 0
for row in infile:
    sent_id = row['predicate']['event_id']
    sentence = row['text']
    predicate = row['predicate']['lemma']
    predicate_role = row['predicate']['event_type']
    # TODO: for now I am only predicting for existing arguments!
    for arg in row['arguments']:
        arg_role = arg['role_type']
        ans_str = arg["text"]
        ans_span = arg["span"]
        ans_span = ans_span.split(':')
        ans_span = [[int(ans_span[0]), int(ans_span[1])]]
        question = qs[counter]
        predictions = arg_dict[sent_id+predicate+arg_role]
        predictions = [pred['sentence'] for pred in predictions]
        counter += 1
        outfile.writerow({'sentence':sentence, 'predicate':predicate, 'event_type':predicate_role, 'event_id':sent_id, 'gold_argument':ans_str, 'gold_span': ans_span, 'role_type':arg_role, 'predicted_arguments':'%%%'.join(predictions), 'arg_question':question})


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

