import json
import jsonlines
import pandas as pd
from pathlib import Path
from typing import Union
import csv

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


