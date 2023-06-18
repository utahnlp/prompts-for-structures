import csv
import jsonlines


#{"doc_id": "APW_ENG_20030318.0689", "sent_id": "APW_ENG_20030318.0689-10", "tokens": ["Britain", ",", "Spain", ",", "Denmark", ",", "Italy", ",", "the", "Netherlands", "and", "Portugal", "back", "the", "United", "States", "while", "France", "and", "Germany", "lead", "a", "group", "of", "nations", "opposing", "military", "action", "."], "sentence": "Britain, Spain, Denmark, Italy, the Netherlands and Portugal back the United States while France and Germany lead a group of nations opposing military action.", "event_mentions": [{"event_type": "Conflict:Attack", "trigger": {"text": "action", "start": 27, "end": 28}, "arguments": [{"text": "military", "role": "Attacker", "start": 26, "end": 27}]}]}

infile = csv.DictReader(open(''))
