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
    row.append(pred)
    outfile.writerow(row)

out = []
gold_list = []
# with open(f"filtered_gens_vero.bin", "wb") as outfile:
#     for key, value in arg_dict.items():
#         gold_list.append(gold_dict[key])
#         out.append(value)
#     pickle.dump(out, outfile)
#
# with open(f"gold_vero.bin", "wb") as outfile:
#     pickle.dump(gold_list, outfile)

def sort_list(mylist):
    save = {}
    probs = []
    for item in mylist:
        probs.append(item["score"])
        save[item["score"]] = item["sentence"]
    probs.sort(reverse=True)
    out = []
    for prob in probs:
        out.append({"sentence": save[prob], "score": prob})
    return out

infile = jsonlines.open("/Users/valentinapy/PycharmProjects/prompts-for-structures/data/test.event.json")
avg_counts = []
for row in infile:
    sent_id = row['sent_id']
    sentence = ' '.join(row['tokens'])
    event_mentions = row["event_mentions"]
    if len(event_mentions)>0:
        for event_mention in event_mentions:
            arguments = event_mention["arguments"]
            if len(arguments)>0:
                predicate = event_mention['trigger']['text']
                avg_counts.append(len(arguments))
                for arg in arguments:
                    arg_role = arg["role"]
                    gold_arg = arg["text"]
                    gold_span = str(arg["start"])+':'+str(arg["end"])
                    if sent_id+predicate+arg_role in arg_dict:
                        sorted = sort_list(arg_dict[sent_id+predicate+arg_role])
                        out.append(sorted)
                    else:
                        out.append([{"sentence": 'None', "score": 1.0}])
                    print(out[-1])
                    print(gold_arg)
                    gold_list.append(gold_arg)
print(sum(avg_counts)/len(avg_counts))
with open(f"../../../data/filtered_gens_vero.bin", "wb") as outfile:
    pickle.dump(out, outfile)
with open(f"../../../data/gold_vero.bin", "wb") as outfile:
    pickle.dump(gold_list, outfile)