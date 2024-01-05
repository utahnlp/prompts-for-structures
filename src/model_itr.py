from itr_prompts import construct_itr_prompt_srl, construct_itr_prompt_coref
from evaluate import evaluate
from graph import construct_graph, Graph
from utils import Config, analyse_beams, restrict_vocab
from plot import plot_yes_no, plot_calibration, plot_score_diff
from preprocess import preprocess_file
from prompts import generate_prompts
from inference import run_inference

import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import Union

GPU_ID = "0"
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2121)


class PromptModel():
    def __init__(self, config: Config):
        """ Constructor for the Prompt Model Pipeline. Preprocesses the data 
        to break it down into components of a structure. Each row essentially
        should have the essential ingredients for constructing the prompt.
        Also, intiializes the prompt model.
        Inputs
        -----------------------
        config- utils.Config. A config dictionary which loads the meta-data and 
                paramaeters from the confg.ini file. 
        """
        self.config = config
        self.data = preprocess_file(
                    file_path = Path(self.config.data_dir, self.config.data_file),
                    task = self.config.task_name,
                    dataset = self.config.dataset_name
                )
        
        print(f"Total number of queries: {len(self.data)}")        
        self.init_model(self.config.model)



    def init_model(self, model_name: str):
        """ Initialize tokenizers and models Initialize tokenizers and models.
        Models currently supported - "flan-t5-xl"
        """
        if model_name in ["flan-t5-xl"]:
            self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
            self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl").to(device)
        else:
            raise ValueError("Only Flan models supported for iterative prompting experiments")



    def generate_prompts_itr(self, pr_type):
        """ Construct prompt for all the sentences in the dataset
        Inputs
        ---------------
        pr_type: str. Values in {"wo_ans_string", "with_ans_string"}. If set to "with_ans_string", prompt
                    will have empty slots for future answers which can be filled in the generation phase. These
                    answers can be used to ask the model to avoid overlapping answers. 

        Outputs
        --------------
        prompts: List[str]. List of prompts input to the model
        gold: List[str]. List of gold answers. Parallel list to `prompts`

        """
        prompts = []
        gold = []
        predicate = None

        if self.config.task_name in ["srl"]:
            pred_questions = []
            for ix, row in self.data.iterrows():
                # Intializing running parameters 
                if predicate == None:
                    predicate = row['predicate']
                    sent_id = row["sent_id"]
                    sentence = row["sentence"]
                # Marking the end of the structure and updating the running parameters
                if (predicate != row["predicate"]) or (sent_id != row["sent_id"]):
                    predicate = row['predicate']
                    sent_id = row["sent_id"]
                    sentence = row["sentence"]
                    pred_questions = []

                pr = construct_itr_prompt_srl(row["sentence"], row["question"], pred_questions, pr_type)
                prompts.append(pr)
                gold.append(row["answer"]) 
                pred_questions.append(row['question'])
        else:
            doc_id = None
            pred_questions = []
            for ix, row in self.data.iterrows():
                if predicate == None:
                    doc_id = row['doc_id']

                if (doc_id != row["doc_id"]):
                    doc_id = row["doc_id"]
                    sentence = row["sentence"]
                    pred_questions = []

                pr = construct_itr_prompt_coref(row["sentence"], row["entity1"], row["entity2"], pred_questions)
                prompts.append(pr)
                gold.append(row["answer"]) 
                pred_questions.append([row["entity1"], row["entity2"]])

        return prompts, gold



    def generate(self, beam_size=1, test_mode=False):
        """ Method to prepare prompts and generate.
        Outputs
        ============
        prompts: List[str]. List of prompts input to the model
        gold: List[str]. List of gold answers. Parallel list to `prompts`
        generation: List[List[dict]]. Contains the outputs generated from the model. Each outer list element
                    corresponds to a list of answers for a prompt. The inner list is arranged in descending
                    order of score. Each list element is a dictionary with the keys "sentence" that contains 
                    a candidate answer string as value, and "score" contain the sequence score for that answer
        """
        pr_type = "wo_ans_string"
        prompts, gold = self.generate_prompts_itr(pr_type)    # Generate prompts and their answers)    
        generation = [] # The list contains all the generation from the model

        ####### Paramters for generation
        restriction, max_len, calib_prompt = restrict_vocab(self.config) # Restrictions on generation vocabulary and dummy prompts

        def restrict_decode_vocab(batch_idx, prefix_beam):
                """ Function to restrict decode vocab to some tokens. Source: https://github.com/huggingface/transformers/issues/15169
                """
                return self.tokenizer(restriction, add_special_tokens=True, return_tensors="pt", is_split_into_words=True)['input_ids'].tolist()

        if restriction != None:
            calib_order = restriction.copy()


        # Iterate over prompts
        predicate = None
        pred_answers = []
        for ix, row in tqdm(self.data.iterrows()):
            if self.config.task_name == "srl":
                # Initializing running parameters for the structure
                if predicate == None:
                    predicate = row['predicate']
                    sent_id = row["sent_id"]
                    sentence = row["sentence"]
                # Indicates that a change of structure/predicate; resetting the pred_answeres
                # and update the running parameters
                if (predicate != row["predicate"]) or (sent_id != row["sent_id"]):
                    predicate = row['predicate']
                    sent_id = row["sent_id"]
                    sentence = row["sentence"]
                    pred_answers = []
            else:
                if predicate == None:
                    doc_id = row["doc_id"]
                if (doc_id != row["doc_id"]):
                    doc_id = row["doc_id"]
                    pred_answers = []
                    print(f"\n\n\nStructure chnage")

            # Filling the prompt up with answers discovered in the structure
            prompt = prompts[ix]
            if len(pred_answers) != 0:
                if pr_type == "with_ans_string":
                    doub_list = pred_answers + pred_answers
                    prompt = prompt.format(*doub_list)
                else:
                    prompt = prompt.format(*pred_answers)
        
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            
            with torch.no_grad():
                if self.config.task_name in ['coref']:
                    outputs = self.model.generate(input_ids.to(device), num_return_sequences=beam_size, num_beams=beam_size, output_scores=True, return_dict_in_generate=True, prefix_allowed_tokens_fn= restrict_decode_vocab, max_length=max_len)
                else:
                    outputs = self.model.generate(input_ids.to(device), num_return_sequences=beam_size, num_beams=beam_size, output_scores=True, return_dict_in_generate=True)

            if test_mode:
                if ix == 76:
                    break

            # Storing top 10 sequences along with their scores 
            prompt_gens = []
            # For tasks like coref the only valid answers would be Yer or No
            # The lists tally the valid answers against their respective scores 
            if self.config.task_name in ['coref']:
                val_answers = calib_order.copy()
                ans_scores = [0]*len(val_answers)
            
            # Process outputs into a list of dictionary to store the generation and scores
            passed_ans = []
            for seq_ix in range(beam_size): #Iterate over the beam size
                seq_dict = {}
                sentence = self.tokenizer.decode(outputs.sequences[seq_ix], skip_special_tokens=True).replace(",", " ,")
                score = outputs.sequences_scores[seq_ix].item()
                
                # In the case of coreference and some other tasks the outputs are restrictive
                # based on the 'val_answers' list. We post-process these task outputs separately
                if self.config.task_name in ['coref']:
                    # Only consider answers in the valid answer list and ones thta have not been generated yet
                    if (sentence in val_answers) and (sentence not in passed_ans):
                        ans_scores[val_answers.index(sentence)] = score
                        if sentence not in passed_ans:
                            passed_ans.append(sentence)
                    if len(passed_ans) == len(val_answers):
                        break
                else:
                    prompt_gens.append({"sentence": sentence, "score":score})   # Save generation and score
            
            # For coref and some other tasks, we need to post process the output and save the 
            # generations in descending order of score. If needed these scores might have to be calibrated
            if self.config.task_name in ['coref']: #and self.config.model!='macaw-3b':
                if self.config.score_type == "prob":
                    out_sf = torch.softmax(torch.Tensor(ans_scores),0)  #Change to probability
                else:
                    out_sf = torch.Tensor(ans_scores)
               
                recalib_scores = out_sf
                
                # Sort by descending score
                temp = sorted(zip(calib_order, recalib_scores.tolist()),key=lambda i:i[1],reverse=True)
                
                # Post-process answers to Yes/No
                for l1, l2 in temp:
                    if l1 in ["True","False"]:
                        if l1 == "True":
                            prompt_gens.append({"sentence":"No", "score":l2})
                        else:
                            prompt_gens.append({"sentence":"Yes", "score":l2})
                    elif l1 in ["$answer$ = Yes", "$answer$ = No"]:
                        if l1 == "$answer$ = No":
                            prompt_gens.append({"sentence":"No", "score":l2})
                        elif l1 == "$answer$ = Yes":
                            prompt_gens.append({"sentence":"Yes", "score":l2})
                    else:
                        prompt_gens.append({"sentence":l1, "score":l2})
                

            generation.append(prompt_gens)
            pred_answers.append(prompt_gens[0]["sentence"])
            torch.cuda.empty_cache()


        return prompts, gold, generation



    def constrained_inference(self, generations, sanity_check=False, meta= None):
        """ Constrained Inference Module
        """
        const_ans = run_inference(self.config.task_name, self.data, generations, sanity_check, meta)

        return const_ans


def add_parser_args(parser):
    parser.add_argument('--config_file', default= "config.ini", type=str)
    parser.add_argument('--read_generated', action='store_true')
    parser.add_argument('--read_inferences', action='store_true')
    return parser


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser = add_parser_args(parser)
    args = vars(parser.parse_args())
    
    config = Config(filename=args['config_file'])
    model = PromptModel(config)
    
    # Intermediary dump data
    task_name = config.task_name           #"srl"#"coref"
    dataset_name = config.dataset_name                         #"qasrl2"     #"ecbp"
    model_name = config.model                           #"t53b"               #"macaw3b"
    read_spec = config.read_spec            #"highlight_fullcontext"
    spec_det = config.spec_det              #"highlight_fullcontext_rtol"
    read_file_infix = f"{model_name}{read_spec}"
    file_infix = f"{model_name}{spec_det}"
    
    run_generate = not args['read_generated']
    run_inference_module = not args['read_inferences']

    ####### STEP 1. Generation
    ### Generate & dump generations and gold
    if run_generate:
        _, gold, gens = model.generate(beam_size=20, test_mode=False) 
        with open(f"./../dumps/{dataset_name}_{task_name}_{file_infix}_gens.bin","wb") as out:
            pickle.dump(gens, out)
        with open(f"./../dumps/{dataset_name}_{task_name}_{file_infix}_gold","wb") as out:
            pickle.dump(gold, out) 
    ### Read Dumps
    else:
        with open(f"./../dumps/{dataset_name}_{task_name}_{read_file_infix}_gens.bin","rb") as out:
            gens = pickle.load(out)
        with open(f"./../dumps/{dataset_name}_{task_name}_{read_file_infix}_gold","rb") as out:
            gold = pickle.load(out)
   

    ######## STEP 2. Running Inference
    if run_inference_module:
        meta= {'thresh': 0.5, "config": model.config}
        const_ans = model.constrained_inference(gens, sanity_check=False, meta= meta)
        with open(f"./../dumps/{dataset_name}_{task_name}_{file_infix}_consans.bin","wb") as out:
            pickle.dump(const_ans, out)
    else:
        with open(f"./../dumps/{dataset_name}_{task_name}_{read_file_infix}_consans.bin","rb") as out:
            const_ans = pickle.load(out)
    
    
    ######### STEP 3. Evaluation
    if config.task_name == "coref":
        # Baseline Coref
        print("Single Cluster Baseline")
        meta = {"gold_dump_file": f"./../results/coref/{dataset_name}_{file_infix}_gold.txt", "pred_dump_file": f"./../results/coref/{dataset_name}_{file_infix}_yes.txt", "constrained":True}
        yes_gens = ["Yes"]*len(gold)   #Taking the top path in the beam
        evaluate(model.data, model.config, yes_gens, meta)

        print("Singleton Baseline")
        meta = {"gold_dump_file": f"./../results/coref/{dataset_name}_{file_infix}_gold.txt", "pred_dump_file": f"./../results/coref/{dataset_name}_{file_infix}_no.txt", "constrained":True}
        no_gens = ["No"]*len(gold)   #Taking the top path in the beam
        evaluate(model.data, model.config, no_gens, meta)


    ## Unconstrained Evaluation
    print("Iterative Prompting with Verbal constraints")
    meta = {"gold_dump_file": f"./../results/coref/{dataset_name}_{file_infix}_gold.txt", "pred_dump_file": f"./../results/coref/{dataset_name}_{file_infix}_uncons.txt", "constrained":False}
    cot_gens = [gen[0]["sentence"] for gen in gens]   #Taking the top path in the beam
    evaluate(model.data, model.config, cot_gens, meta)
    
    ## Iterative Prompting
    print("Iterative Prompting with Verbal Constraints plus Inference")
    meta = {"gold_dump_file": f"./../results/coref/{dataset_name}_{file_infix}_gold.txt", "pred_dump_file": f"./../results/coref/{dataset_name}_{file_infix}_cot.txt", "constrained":False}
    evaluate(model.data, model.config, const_ans, meta)



  
