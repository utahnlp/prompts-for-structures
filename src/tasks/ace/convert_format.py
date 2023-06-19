import csv
import jsonlines
import pickle
from collections import defaultdict
#{"doc_id": "APW_ENG_20030318.0689", "sent_id": "APW_ENG_20030318.0689-10", "tokens": ["Britain", ",", "Spain", ",", "Denmark", ",", "Italy", ",", "the", "Netherlands", "and", "Portugal", "back", "the", "United", "States", "while", "France", "and", "Germany", "lead", "a", "group", "of", "nations", "opposing", "military", "action", "."], "sentence": "Britain, Spain, Denmark, Italy, the Netherlands and Portugal back the United States while France and Germany lead a group of nations opposing military action.", "event_mentions": [{"event_type": "Conflict:Attack", "trigger": {"text": "action", "start": 27, "end": 28}, "arguments": [{"text": "military", "role": "Attacker", "start": 26, "end": 27}]}]}

with open(f"/Users/valentinapy/PycharmProjects/prompts-for-structures/data/dumps_vero/dumps/ace_ace_t5-3bacevero_gens.bin", "rb") as out:
    gens = pickle.load(out)
with open(f"/Users/valentinapy/PycharmProjects/prompts-for-structures/data/dumps_vero/dumps/ace_ace_t5-3bacevero_gold", "rb") as out:
    gold = pickle.load(out)

with open(f"/Users/valentinapy/PycharmProjects/prompts-for-structures/data/dumps_vero/dumps/ace_ace_t5-3bacevero_consans.bin", "rb") as out:
    const_ans = pickle.load(out)

uncon_gens = [gen[0]["sentence"] for gen in gens]

infile = csv.reader(open('/Users/valentinapy/PycharmProjects/prompts-for-structures/data/dumps_vero/dumps/argument_questions.csv'))

outfile = jsonlines.open('gold.jsonlines', 'w')

sent_dict = defaultdict(lambda : {})
for uncon_gen, row in zip(uncon_gens, infile):
    out = {}
    out["sent_id"] = row[4]
    out["doc_id"] = row[4].split('-')[0]
    out["sentence"] = row[9]
    out["tokens"] = row[9].split()
    out["event_mentions"] = []
    event_dict = {}
    event_dict["event_tyoe"] = ':'.join(row[8].split('_'))
    event_dict["trigger"] = {}
    event_dict["trigger"]["text"] = row[2]
    event_dict["trigger"]["start"] = int(row[3].split(':')[0])
    event_dict["trigger"]["end"] = int(row[3].split(':')[1])
    event_dict["arguments"] = []
    arg_dict = {}
    arg_dict["text"] = uncon_gen
    arg_dict["role"] = row[7]
    sentence = row[9]
    print('new')
    print(uncon_gen)
    print(sentence)
    try:
        char_s = sentence.index(uncon_gen)
        pre_sent = len(sentence[:char_s].split())
        post_sent = len(sentence[:char_s+len(uncon_gen)].split())
    except ValueError:
        print('oh no')
        pre_sent = 0
        post_sent = 0
    print(sentence.split()[pre_sent:post_sent])
    print(uncon_gen)
    arg_dict["start"] = pre_sent
    arg_dict["end"] = post_sent
    event_dict["arguments"].append(arg_dict)
    out["event_mentions"].append(event_dict)