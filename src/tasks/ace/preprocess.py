import json
import jsonlines
import pandas as pd
from pathlib import Path
from typing import Union
import csv
from collections import defaultdict
import sys

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
    infile = csv.reader(open('../data/questions/type_question_map.tsv'), delimiter='\t')

    for row in infile:
        if not 'Time' in row[0]:
            predicate, arg = row[0].split('_')
            predicate = '_'.join(predicate.split('.'))
            q_dict[predicate][arg] = row[1]

    return q_dict

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
    infile = csv.DictReader(open(filepath), delimiter='\t')
    processed_data = []
    for row in infile:
        sent_id = row['arg_id']
        sentence = row['text']
        predicate = row['predicate_lemma']
        ques_str = row["query_question"]
        ans_str = row["argument_text"]
        ans_span = row["argument_span"]
        ans_span = ans_span.split(':')
        ans_span = [[int(ans_span[0]),int(ans_span[1])]]
        processed_data.append(
            [sent_id, sentence, predicate, ques_str, ans_str, ans_span])

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
    infile = jsonlines.open(filepath)
    processed_data = []
    outfile = csv.writer(open('missing.csv', 'w'))
    for row in infile:
        sent_id = row['predicate']['event_id']
        sentence = row['text']
        predicate = row['predicate']['lemma']
        predicate_role = row['predicate']['event_type']
        #TODO: for now I am only predicting for existing arguments!
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
                ans_span = [[int(ans_span[0]),int(ans_span[1])]]
                processed_data.append(
                    [sent_id, sentence, predicate, ques_str, ans_str, ans_span])

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
    processed_data = []
    for row in infile:
        sent_id = row['arg_id']
        sentence = row['text']
        predicate = row['predicate_lemma']
        ques_str = row["query_question"]
        ans_str = row["argument_text"]
        ans_span = row["argument_span"]
        ans_span = ans_span.split(':')
        ans_span = [[int(ans_span[0]),int(ans_span[1])]]
        processed_data.append(
            [sent_id, sentence, predicate, ques_str, ans_str, ans_span])

    columns = ["sent_id", "sentence", "predicate", "question", "answer", "ans_span"]
    data_df = pd.DataFrame(processed_data, columns=columns)

    return data_df


