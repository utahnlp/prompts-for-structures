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

#problematic_ix = [709,710,711,1219,1220,1221,1502,1503,2694,2695,2696,2697,2921,2922,2923,3263,3264,3265,3266,3267,3390,3405,3406,3459,3460,4214,4215,4216,4508,4509,4510]
#problematic_ix = [4145,4146,4147,4457,4458,4459,4460,5856,5857,5858,5926,5927,5928]


class PromptModel():
    def __init__(self, config: Config):
        self.config = config
        self.data = preprocess_file(
                    file_path = Path(self.config.data_dir, self.config.data_file),
                    task = self.config.task_name,
                    dataset = self.config.dataset_name
                )
        print(len(self.data))
        #print(self.data["doc_id"].head(80))
        #exit()
        self.init_model(self.config.model)


    def init_model(self, model_name: str):
        """ Initialize tokenizers and models Initialize tokenizers and models  
        """
        if model_name == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
            self.model = T5ForConditionalGeneration.from_pretrained("t5-large").to(device)
        if model_name == "t5-11b":
            self.tokenizer = T5Tokenizer.from_pretrained("t5-11b")
            self.model = T5ForConditionalGeneration.from_pretrained("t5-11b",use_cdn = False).to(device)
        if model_name == "t5-3b":
            self.tokenizer = T5Tokenizer.from_pretrained("t5-3b")
            self.model = T5ForConditionalGeneration.from_pretrained("t5-3b").to(device)
        elif model_name == "unified-qa":
            self.tokenizer = T5Tokenizer.from_pretrained("allenai/unifiedqa-t5-large")
            self.model = T5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-t5-large").to(device)



    def calibrate(self, beam_size):
        if self.config.task_name in ['coref']:
            def restrict_decode_vocab(batch_idx, prefix_beam):
                return self.tokenizer(["Yes","No"], add_special_tokens=True, return_tensors="pt", is_split_into_words=True)['input_ids'].tolist()
       
            calib_out = self.model.generate( self.tokenizer("Yes or No?", return_tensors="pt").input_ids.to(device), num_return_sequences=beam_size, num_beams=beam_size, output_scores=True, return_dict_in_generate=True, prefix_allowed_tokens_fn= restrict_decode_vocab, max_length=2)
            gens = []
            ans_scores= []
            ans_order = []
            val_answers = ["Yes","No"]
            for seq_ix in range(beam_size):
                seq_dict = {}
                seq_dict["sentence"] = self.tokenizer.decode(calib_out.sequences[seq_ix], skip_special_tokens=True).replace(",", " ,")
                seq_dict["score"] = calib_out.sequences_scores[seq_ix].item()
                if seq_dict["sentence"] in val_answers:
                    ans_order.append(seq_dict["sentence"])
                    ans_scores.append(seq_dict["score"])
                    val_answers.remove(seq_dict["sentence"])
                    
                gens.append(seq_dict)
            p_cf = 1/torch.softmax(torch.Tensor(ans_scores),0)
            
            return p_cf, ans_order
        else:
            return None




    def generate(self, beam_size=1, test_mode=False):
        """ Method to prepare prompts and generate.
        """
        prompts, gold = generate_prompts(self.data, self.config)    # Generate prompts and their answers
        generation = []
        def restrict_decode_vocab(batch_idx, prefix_beam):
            return self.tokenizer(["Yes","No"], add_special_tokens=True, return_tensors="pt", is_split_into_words=True)['input_ids'].tolist()
       
        calib_out, calib_order = self.calibrate(beam_size)
        print(calib_out)
        print(calib_order)
        #calib_order = ["Yes","No"]
        #exit()

        for ix, prompt in tqdm(enumerate(prompts)):
            #if ix not in problematic_ix:
            #    continue
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            
            with torch.no_grad():
                if self.config.task_name in ['coref']:
                    outputs = self.model.generate(input_ids.to(device), num_return_sequences=beam_size, num_beams=beam_size, output_scores=True, return_dict_in_generate=True, prefix_allowed_tokens_fn= restrict_decode_vocab, max_length=2)
                else:
                    outputs = self.model.generate(input_ids.to(device), num_return_sequences=beam_size, num_beams=beam_size, output_scores=True, return_dict_in_generate=True)


            if test_mode:
                if ix == 76:
                    break

            # Storing top 10 sequences along with their scores 
            prompt_gens = []
            # For tasks like coref the only valid answers would be Yer or No
            # The lists 
            if self.config.task_name in ['coref']:
                val_answers = calib_order.copy()
                ans_scores = [0]*len(val_answers)
            
            passed_ans = []
            for seq_ix in range(beam_size):
                seq_dict = {}
                sentence = self.tokenizer.decode(outputs.sequences[seq_ix], skip_special_tokens=True).replace(",", " ,")
                score = outputs.sequences_scores[seq_ix].item()
                
                if self.config.task_name in ['coref']:
                    if (sentence in val_answers) and (sentence not in passed_ans):
                        ans_scores[val_answers.index(sentence)] = score
                        if sentence not in passed_ans:
                            passed_ans.append(sentence)
                    if len(passed_ans) == len(val_answers):
                        break
                else:
                    prompt_gens.append({"sentence": sentence, "score":score})
            
            if self.config.task_name in ['coref']:
                #out_sf = torch.softmax(torch.Tensor(ans_scores),0)
                #recalib_scores = calib_out*out_sf
                
                recalib_scores = torch.Tensor(ans_scores)
                temp = sorted(zip(calib_order, recalib_scores.tolist()),key=lambda i:i[1],reverse=True)
                
                for l1, l2 in temp:
                    prompt_gens.append({"sentence":l1, "score":l2})
        
            generation.append(prompt_gens)

        return prompts, gold, generation



    def constrained_inference(self, generations, sanity_check=False):
        """ Constrained Inference Module
        """
        predicate = None
        sentence = None
        sent_id = None
        pred_gens = []
        const_ans = []
        gold_ans = []
        gold_ans_spans = []
        invalid_gold = 0

        cnt_ix = -1
        for ix, row in tqdm(self.data.iterrows()):
            #if ix not in problematic_ix:
            #    continue
            cnt_ix += 1
            if predicate == None:
                predicate = row["predicate"]
                sentence = row["sentence"]
                sent_id = row["sent_id"]

            if ((predicate != row["predicate"]) or (sent_id != row["sent_id"])):
                predicate = row["predicate"]
                sent_id = row["sent_id"]
                a_span = None
                c_ans, g_inv = construct_graph(sentence, pred_gens,ix,gold_ans, sanity_check, ans_span=gold_ans_spans)
                const_ans += c_ans
                invalid_gold += g_inv

                sentence = row["sentence"]
                pred_gens = []
                gold_ans = []
                gold_ans_spans = []
                    
            pred_gens.append(generations[cnt_ix])
            gold_ans.append(row["answer"])
            if "ans_span" in row.keys():
                gold_ans_spans.append(row["ans_span"])
           
        c_ans, g_inv = construct_graph(sentence, pred_gens,len(self.data), gold_ans, sanity_check)
        const_ans += c_ans
        invalid_gold += g_inv
        print(f"# Gold answers not perfect sub-sequences: {invalid_gold}")

        return const_ans



if __name__ == "__main__":
    config = Config()
    model = PromptModel(config)

    _, gold, gens = model.generate(beam_size=20, test_mode=True)
    #for ix in range(50):
    #    print(gold[ix])
    #    print(gens[ix][0]["sentence"])
    #    print()
    #print(gens[0])
    ##exit()
    ##with open("outputc_crefecbp_t53b.bin", "wb") as output:
    ##    pickle.dump(gens, output)
    
    #with open("output.bin","rb") as data:
    #    gens = pickle.load(data)

    #with open("output_qasrl2.bin","rb") as data:
    #    gens = pickle.load(data)
    ##const_ans = model.constrained_inference(gens, sanity_check=False)
    
    ##with open("pred_wiki_tr_t53b.bin","wb") as output:
    ##    pickle.dump(const_ans, output)
    #analyse_beams(gold, gens, root_analysis=True)
    #with open("pred_qasrl2_dev.bin","rb") as output:
    #    const_ans = pickle.load(output)

    ## Unconstrained Evaluation
    uncon_gens = [gen[0]["sentence"] for gen in gens]
    evaluate(model.data, model.config, uncon_gens)
    exit()
    print("Constrained")
    evaluate(model.data, model.config, const_ans)


