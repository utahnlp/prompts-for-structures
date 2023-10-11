


def construct_cot_prompt_srl(sent, ques, prev_ques, pr_type="wo_ans_string"):
    if len(prev_ques) == 0: 
        if type(sent) != list:
            return f"""{sent} \n In the above sentence, {ques}"""
        else:
            return f"""{" ".join(sent)} \n In the above sentence, {ques}"""
    else:
        if type(sent) != list:
            inst = f"For the sentence given below, answer the following questions such that the answers do not overlap.\n\n Sentence: {sent}"
        else:
            inst = f"""For the sentence given below, answer the following questions such that the answers do not overlap.\n\n Sentence: {" ".join(sent)}"""

        for q in prev_ques:
            inst = inst + f"\nQuestion: {q}\n"
            inst = inst + "Answer: {}\n"
        if pr_type == "wo_ans_string":
            inst = inst + f"\nQuestion: {ques} Make sure the answer is present in the sentence.\nAnswer: "
        elif pr_type == "with_ans_string":
            inst = inst + f"\nQuestion: {ques} Make sure the answer is present in the sentence. The answer should not be in"
            for ques_i in range(len(prev_ques)-1):
                inst = inst + " \"{}\","
            if len(prev_ques) >1:
                inst = inst + " or \"{}\"."
            else:
                inst = inst + " \"{}\"."
            inst = inst + "\nAnswer: "
 
        return inst



def construct_cot_prompt_coref(sent, ent1, ent2, prev_ques):
    if len(prev_ques) == 0:
        return f"""{sent} \n In the above passage, does {ent1} refer to {ent2}? Yes or No?"""
    else:
        inst = f"For the passage below, you will be given a series of Yes/No questions. Answer the question such that the answers are consistent with the examples given below.\n\n Passage: {sent}"
        for q in prev_ques:
            inst = inst + f"\nQuestion: Does {q[0]} refer to {q[1]}? Yes or No?\n"
            inst = inst + "Answer: {}\n"
        inst = inst + f"\nQuestion:  Does {ent1} refer to {ent2}? Yes or No?\nAnswer: "

        return inst


