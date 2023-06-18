import json
import jsonlines
import pandas as pd
from pathlib import Path
from typing import Union
import csv
from collections import defaultdict
import sys
import re

def read_type_questions():
    #type : question
    type_q_map = defaultdict(lambda : '')
    infile = csv.DictReader(open('../data/questions/questions.tsv'), delimiter='\t')

    for row in infile:
        type_q_map[row['type']]=row['question']

    return type_q_map

def read_type_questions_yesno():
    #type : question
    type_q_map = defaultdict(lambda : '')
    infile = csv.DictReader(open('../data/questions/questions.tsv'), delimiter='\t')

    for row in infile:
        type_q_map[row['type']]=row['question2']

    return type_q_map

def read_types():
    # predicate : argument : types
    type_dict = defaultdict(lambda: defaultdict(lambda: []))
    infile = csv.reader(open('../data/questions/type_question_map2.tsv'), delimiter='\t')

    for row in infile:
        #if not 'Time' in row[0]:
        predicate, arg = row[0].split('_')
        predicate = '_'.join(predicate.split('.'))
        type_dict[predicate][arg] = row[2:]

    return type_dict

def read_questions():
    # predicate : argument : question
    q_dict = defaultdict(lambda : defaultdict(lambda : []))
    infile = csv.reader(open('../data/questions/type_question_map2.tsv'), delimiter='\t')

    for row in infile:
        #if not 'Time' in row[0]:
        predicate, arg = row[0].split('_')
        predicate = '_'.join(predicate.split('.'))
        q_dict[predicate][arg].append(row[1])

    return q_dict

def preprocess_ace_types_yesno(filepath: Union[str, Path]) -> pd.DataFrame:
    """ Preprocessing function for ACE with questions.
    Input
    ----------------------
    filepath: str or pathlib.Path. Input data file path

    Output
    ----------------------
    data_df: pd.DataFrame. Dataframe where each row represents a row
            for prompting.
    """
    type_q_dict = read_type_questions_yesno()
    # predicate : argument : types
    q_dict = read_types()
    infile = csv.DictReader(open(filepath), delimiter=',')
    processed_data = []
    outfile = csv.writer(open('type_questions_yesno.csv', 'w'))
    for row in infile:
        sent_id = row['event_id']
        sentence = row['sentence']
        predicate = row['predicate']
        predicate_role = row['event_type']
        gold_arg = row["gold_argument"]
        gold_span = row["gold_span"]
        #TODO: for now I am only predicting for existing arguments!
        predicted_arguments = row["predicted_arguments"].split('%%%')
        arg_role = row["role_type"]
        for arg in predicted_arguments:
            ans_str = arg
            if predicate_role in q_dict:
                if arg_role in q_dict[predicate_role]:
                    types = q_dict[predicate_role][arg_role]
            for type in types:
                if type in type_q_dict:
                    ques_str = type_q_dict[type]
                    print(ques_str)
                    processed_data.append(
                        [sent_id, sentence, predicate, ques_str, ans_str])
                    outfile.writerow([ques_str, arg, gold_arg, gold_span, sent_id, predicate, type, arg_role, predicate_role, sentence])

    columns = ["sent_id", "sentence", "predicate", "question", "answer"]
    data_df = pd.DataFrame(processed_data, columns=columns)

    return data_df


def preprocess_ace_types(filepath: Union[str, Path]) -> pd.DataFrame:
    """ Preprocessing function for ACE with questions.
    Input
    ----------------------
    filepath: str or pathlib.Path. Input data file path

    Output
    ----------------------
    data_df: pd.DataFrame. Dataframe where each row represents a row
            for prompting.
    """
    type_q_dict = read_type_questions()
    # predicate : argument : types
    q_dict = read_types()
    infile = jsonlines.open(filepath)
    processed_data = []
    outfile = csv.writer(open('type_questions.csv', 'w'))
    for row in infile:
        sent_id = row['predicate']['event_id']
        sentence = row['text']
        predicate = row['predicate']['lemma']
        predicate_role = row['predicate']['event_type']
        #TODO: for now I am only predicting for existing arguments!
        for arg in row['arguments']:
            arg_role = arg['role_type']
            # if 'Time' in arg_role:
            #     pass
            # else:
            if predicate_role in q_dict:
                if arg_role in q_dict[predicate_role]:
                    types = q_dict[predicate_role][arg_role]
                else:
                    print(predicate_role)
                    print(arg_role)
            else:
                print(predicate_role)
                print(arg_role)
            ans_str = arg["text"]
            ans_span = arg["span"]
            ans_span = ans_span.split(':')
            ans_span = [[int(ans_span[0]), int(ans_span[1])]]
            for type in types:
                if type in type_q_dict:
                    ques_str = type_q_dict[type]
                    print(ques_str)
                    processed_data.append(
                        [sent_id, sentence, predicate, ques_str, ans_str, ans_span])
                    outfile.writerow([ques_str, sent_id, predicate, type, arg_role])

    columns = ["sent_id", "sentence", "predicate", "question", "answer", "ans_span"]
    data_df = pd.DataFrame(processed_data, columns=columns)

    return data_df

def preprocess_ace_questions(filepath: Union[str, Path]) -> pd.DataFrame:
    """ Preprocessing function for ACE with questions.
    Input
    ----------------------
    filepath: str or pathlib.Path. Input data file path

    Output
    ----------------------
    data_df: pd.DataFrame. Dataframe where each row represents a row
            for prompting.
    """
    # predicate : argument : question
    q_dict = read_questions()
    print(q_dict)
    infile = jsonlines.open(filepath)
    processed_data = []
    outfile = csv.writer(open('/home/valentinap/workspace/prompts-for-structures/dumps/argument_questions.csv', 'w'))
    covered = {}
    for row in infile:
        sent_id = row['predicate']['event_id']
        sentence = row['text']
        predicate = row['predicate']['lemma']
        predicate_string = row['predicate']['text']
        predicate_role = row['predicate']['event_type']
        #TODO: for now I am only predicting for existing arguments!
        for arg in row['arguments']:
            arg_role = arg['role_type']
            # if 'Time' in arg_role:
            #     pass
            # else:
            if predicate_role in q_dict:
                if arg_role in q_dict[predicate_role]:
                    if sent_id+predicate+arg_role in covered:
                        question = q_dict[predicate_role][arg_role][-1]
                    else:
                        question = q_dict[predicate_role][arg_role][0]
                else:
                    print(predicate_role)
                    print(arg_role)
            else:
                print(predicate_role)
                print(arg_role)
            ques_str = question
            if ques_str.startswith('Where does the event'):
                ques_str = re.sub('event', predicate, ques_str)
            if ques_str.startswith('When does it'):
                ques_str = re.sub('it', 'the '+predicate, ques_str)
            ques_str = re.sub('the event', 'the ' + predicate, ques_str)
            #sentence = sentence + ' The event the question is asking about is: '+predicate_string
            #ques_str = re.sub('\?', ' of the'+predicate+'?', ques_str)
            #ques_str = "given the predicate: " + row["text"] + " " + ques_str
            #ques_str = ques_str + ' ' + 'in ' + predicate_string
            #ques_str = ques_str[:-1]+' of '+row["text"]+'?'
            ans_str = arg["text"]
            ans_span = arg["span"]
            ans_span = ans_span.split(':')
            ans_span = [[int(ans_span[0]),int(ans_span[1])]]
            covered[sent_id+predicate+arg_role]=True
            processed_data.append(
                [sent_id, sentence, predicate, ques_str, ans_str, ans_span])
            outfile.writerow([ques_str, sent_id, predicate, predicate_role, arg_role])

    columns = ["sent_id", "sentence", "predicate", "question", "answer", "ans_span"]
    data_df = pd.DataFrame(processed_data, columns=columns)

    return data_df



def preprocess_ace(filepath: Union[str, Path]) -> pd.DataFrame:
    """ Preprocessing function for ACE with questions.
    Input
    ----------------------
    filepath: str or pathlib.Path. Input data file path

    Output
    ----------------------
    data_df: pd.DataFrame. Dataframe where each row represents a row
            for prompting.
    """
    infile = csv.DictReader(open(filepath), delimiter='\t')
    q_dict = read_questions()
    processed_data = []
    for row in infile:
        sent_id = row['arg_id']
        sentence = row['text']
        predicate = row['predicate_lemma']
        ques_str = row["query_question"]
        ques_str = ques_str + ' ' + 'in ' + predicate
        predicate_role = row['event_type']
        predicate_role =  '_'.join(predicate_role.split('.'))
        arg_role = row['role_type']
        if predicate_role in q_dict:
            if arg_role in q_dict[predicate_role]:
                question = q_dict[predicate_role][arg_role][0]
        ques_str = question
        if ques_str.startswith('Where does the event'):
            ques_str = re.sub('event', predicate, ques_str)
        if ques_str.startswith('When does it'):
            ques_str = re.sub('it', 'the ' + predicate, ques_str)
        ques_str = re.sub('the event', 'the ' + predicate, ques_str)
        ans_str = row["argument_text"]
        ans_span = row["argument_span"]
        ans_span = ans_span.split(':')
        ans_span = [[int(ans_span[0]),int(ans_span[1])]]
        processed_data.append(
            [sent_id, sentence, predicate, ques_str, ans_str, ans_span])

    columns = ["sent_id", "sentence", "predicate", "question", "answer", "ans_span"]
    data_df = pd.DataFrame(processed_data, columns=columns)

    return data_df


