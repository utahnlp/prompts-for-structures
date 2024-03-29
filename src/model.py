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
import time

GPU_ID = "1"
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
#device = "cpu"
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
        Models currently supported - "t5-{large,3b,11b}, unified-qa, macaw-3b"
        """
        if model_name == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
            self.model = T5ForConditionalGeneration.from_pretrained("t5-large").to(device)
        elif model_name in ["t5-small","t5-base","t5-3b"]:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        elif model_name in ["t5-11b"]:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=512)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
            #self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
            #self.model = T5ForConditionalGeneration.from_pretrained(model_name, device_map='auto')#.to(device)

            #self.model = T5ForConditionalGeneration.from_pretrained(model_name, load_in_8bit=True, device_map='auto')#.to(device)
        elif model_name in ["flan-t5-xl"]:
            self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
            self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl").to(device)
        elif model_name == "unified-qa":
            self.tokenizer = T5Tokenizer.from_pretrained("allenai/unifiedqa-t5-large")
            self.model = T5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-t5-large").to(device)
        elif model_name[:12] == "unifiedqa-v2":
            self.tokenizer = T5Tokenizer.from_pretrained(f"allenai/{model_name}-1251000")
            self.model = T5ForConditionalGeneration.from_pretrained(f"allenai/{model_name}-1251000").to(device)

        elif model_name in ["macaw-3b","macaw-large","macaw-11b"]:
            self.tokenizer = T5Tokenizer.from_pretrained(f"allenai/{model_name}", model_max_length=512)
            self.model = T5ForConditionalGeneration.from_pretrained(f"allenai/{model_name}").to(device)
            #self.model = T5ForConditionalGeneration.from_pretrained(f"allenai/{model_name}", device_map='balanced_low_0')


    def calibrate(self, beam_size, restrict_ans= ["Yes","No"], max_len = 2, calib_prompt="Yes or No?"):
        if self.config.task_name in ['coref']:
            def restrict_decode_vocab(batch_idx, prefix_beam):
                return self.tokenizer(restrict_ans, add_special_tokens=True, return_tensors="pt", is_split_into_words=True)['input_ids'].tolist()
       
            calib_out = self.model.generate(self.tokenizer(calib_prompt, return_tensors="pt").input_ids.to(device), num_return_sequences=beam_size, num_beams=beam_size, output_scores=True, return_dict_in_generate=True, prefix_allowed_tokens_fn= restrict_decode_vocab, max_length=max_len)
            gens = []
            ans_scores= []
            ans_order = []
            val_answers = restrict_ans.copy()
            for seq_ix in range(beam_size):
                seq_dict = {}
                seq_dict["sentence"] = self.tokenizer.decode(calib_out.sequences[seq_ix], skip_special_tokens=True).replace(",", " ,")
                seq_dict["score"] = calib_out.sequences_scores[seq_ix].item()
                if seq_dict["sentence"] in val_answers:
                    ans_order.append(seq_dict["sentence"])
                    ans_scores.append(seq_dict["score"])
                    val_answers.remove(seq_dict["sentence"])
                    
                gens.append(seq_dict)
        

            if self.config.calibration_type == "calib_before_use": 
                p_cf = 1/torch.softmax(torch.Tensor(ans_scores),0)
            elif self.config.calibration_type == "pmi":
                p_cf = torch.softmax(torch.Tensor(ans_scores),0) 
            elif self.config.calibration_type == "score_diff":
                offset = 0 
                if "Yes" in restrict_ans:
                    for aix, a in enumerate(ans_order):
                        if a == "Yes":
                            offset += ans_scores[aix]
                        elif a == "No":
                            offset -= ans_scores[aix]
                
                p_cf = offset
                ans_order = restrict_ans
            
            return p_cf, ans_order
        else:
            return None, None




    def generate(self, beam_size=1, test_mode=False):
        """ Method to prepare prompts and generate.
        """
        prompts, gold = generate_prompts(self.data, self.config)    # Generate prompts and their answers)    
        do_calibrate = self.config.do_calibrate
        generation = [] # The list contains all the generation from the model

        # This conditions checks for differing priors 
        # This is required since out models model joint probability 
        # instead of conditional. To achieve conditionals, we must subtract 
        # prior probability from the joint. However, if for a structure, these 
        # are negligible, it doesn't matter. This sub-experiment checks for these values
        condition_prob = False
        if condition_prob:
            c_prob_std = []
            curr_id = None

        ####### Paramters for generation
        restriction, max_len, calib_prompt = restrict_vocab(self.config) # Restrictions on generation vocabulary and dummy prompts

        def restrict_decode_vocab(batch_idx, prefix_beam):
                """ Function to restrict decode vocab to some tokens. Source: https://github.com/huggingface/transformers/issues/15169
                """
                return self.tokenizer(restriction, add_special_tokens=True, return_tensors="pt", is_split_into_words=True)['input_ids'].tolist()

        if restriction != None:
            calib_order = restriction.copy()
        # Calibrate if you have a restricted vocabulary
        if do_calibrate:
            calib_out, calib_order = self.calibrate(beam_size, restrict_ans=restriction.copy(), max_len=max_len, calib_prompt=calib_prompt)
            print(f"Calibration Answer Order: {calib_order}")
            print(f"Calibration Scores: {calib_out}")
        

        # Iterate over prompts
        for ix, prompt in tqdm(enumerate(prompts)):
            if condition_prob:
                if curr_id == None:
                    priors = []
                    curr_id = self.data['doc_id'].iloc[ix]
                if curr_id != self.data['doc_id'].iloc[ix]:
                    c_prob_std.append(np.std(priors))
                    priors = []
                    curr_id = self.data['doc_id'].iloc[ix]

            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            
            with torch.no_grad():
                if self.config.task_name in ['coref']:
                    outputs = self.model.generate(input_ids.to(device), num_return_sequences=beam_size, num_beams=beam_size, output_scores=True, return_dict_in_generate=True, prefix_allowed_tokens_fn= restrict_decode_vocab, max_length=max_len)
                    if condition_prob:
                        cond_outs = self.model.generate(input_ids.to(device), return_dict_in_generate=True, output_scores=True)
                        priors.append(np.log(torch.softmax(cond_outs.scores[0],dim=1)[0][self.tokenizer.eos_token_id].cpu().numpy()))
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
                if do_calibrate:
                    # Calibrate
                    if self.config.calibration_type == "calib_before_use":
                        recalib_scores = calib_out*out_sf
                        recalib_scores = torch.softmax(torch.Tensor(recalib_scores),0)
                    elif self.config.calibration_type == "score_diff":
                        if "Yes" in calib_order:
                            out_sf[calib_order.index("Yes")] -=  calib_out
                        recalib_scores = out_sf
                    elif self.config.calibration_type == "pmi":
                        recalib_scores = torch.div(out_sf, calib_out)
                else:
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
            torch.cuda.empty_cache()

        if condition_prob:
            c_prob_std.append(np.std(priors))

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
        start = time.time()
        const_ans = model.constrained_inference(gens, sanity_check=False, meta= meta)
        end = time.time()
        print(f"Inference time: {end-start}")
        exit()
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
    print("Unconstrained")
    meta = {"gold_dump_file": f"./../results/coref/{dataset_name}_{file_infix}_gold.txt", "pred_dump_file": f"./../results/coref/{dataset_name}_{file_infix}_uncons.txt", "constrained":False}
    uncon_gens = [gen[0]["sentence"] for gen in gens]   #Taking the top path in the beam
    evaluate(model.data, model.config, uncon_gens, meta)
    ## Constrained Evaluation
    print("Constrained")
    meta = {"gold_dump_file": f"./../results/coref/{dataset_name}_{file_infix}_gold.txt", "pred_dump_file": f"./../results/coref/{dataset_name}_{file_infix}_cons.txt", "constrained":True}
    evaluate(model.data, model.config, const_ans, meta)


