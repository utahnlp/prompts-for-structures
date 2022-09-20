from evaluate import evaluate
from graph import construct_graph, Graph
from utils import Config, analyse_beams
from preprocess import preprocess_file
from prompts import generate_prompts

import pickle
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import Union

GPU_ID = "1"
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
#device = "cpu"

torch.manual_seed(2121)

problematic_ix = [709,710,711,1219,1220,1221,1502,1503,2694,2695,2696,2697,2921,2922,2923,3263,3264,3265,3266,3267,3390,3405,3406,3459,3460,4214,4215,4216,4508,4509,4510]

class PromptModel():
    def __init__(self, config: Config):
        self.config = config
        self.data = preprocess_file(
                    file_path = Path(self.config.data_dir, self.config.data_file),
                    task = self.config.task_name,
                    dataset = self.config.dataset_name
                )
        self.init_model(self.config.model)


    def init_model(self, model_name: str):
        """ Initialize tokenizers and models Initialize tokenizers and models  
        """
        if model_name == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
            self.model = T5ForConditionalGeneration.from_pretrained("t5-large").to(device)


    def generate(self, beam_size=1, test_mode=False):
        """ Method to prepare prompts and generate.
        """
        prompts, gold = generate_prompts(self.data, self.config)    # Generate prompts and their answers
        generation = []

        for ix, prompt in tqdm(enumerate(prompts)):
            #if ix not in problematic_ix:
            #    continue
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            with torch.no_grad():
                outputs = self.model.generate(input_ids.to(device), num_return_sequences=beam_size, num_beams=beam_size, output_scores=True, return_dict_in_generate=True)
        
            if test_mode:
                if ix == 20:
                    break

            # Storing top 10 sequences along with their scores 
            prompt_gens = []
            for seq_ix in range(beam_size):
                seq_dict = {}
                seq_dict["sentence"] = self.tokenizer.decode(outputs.sequences[seq_ix], skip_special_tokens=True).replace(",", " ,")
                seq_dict["score"] = outputs.sequences_scores[seq_ix].item()
                prompt_gens.append(seq_dict)
                
            generation.append(prompt_gens)

        return prompts, gold, generation



    def constrained_inference(self, generations, sanity_check=False):
        """ Constrained Inference Module
        """
        predicate = None
        sentence = None
        pred_gens = []
        const_ans = []
        gold_ans = []
        invalid_gold = 0

        cnt_ix = -1
        for ix, row in tqdm(self.data.iterrows()):
            #if ix not in problematic_ix:
            #    continue
            cnt_ix += 1
            if predicate == None:
                predicate = row["predicate"]
                sentence = row["sentence"]

            if predicate != row["predicate"]:
                predicate = row["predicate"]
                c_ans, g_inv = construct_graph(sentence, pred_gens,ix,gold_ans, sanity_check)
                const_ans += c_ans
                invalid_gold += g_inv

                sentence = row["sentence"]
                pred_gens = []
                gold_ans = []
                    
            pred_gens.append(generations[cnt_ix])
            gold_ans.append(row["answer"])
           
        c_ans, g_inv = construct_graph(sentence, pred_gens,len(self.data), gold_ans, sanity_check)
        const_ans += c_ans
        invalid_gold += g_inv
        print(f"# Gold answers not perfect sub-sequences: {invalid_gold}")

        return const_ans



if __name__ == "__main__":
    config = Config()
    model = PromptModel(config)
    #_, gold, gens = model.generate(beam_size=20, test_mode=False)
    
    #with open("output.bin", "wb") as output:
    #    pickle.dump(gens, output)
    with open("output.bin","rb") as data:
        gens = pickle.load(data)
    const_ans = model.constrained_inference(gens, sanity_check=False)

    
    #analyse_beams(gold, gens, root_analysis=True)
    
    ## Unconstrained Evaluation
    uncon_gens = [gen[0]["sentence"] for gen in gens]
    evaluate(model.data, model.config, uncon_gens)

    print("Constrained")
    evaluate(model.data, model.config, const_ans)


