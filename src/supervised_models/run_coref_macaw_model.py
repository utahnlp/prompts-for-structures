import argparse
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import f1_score
import torch
from transformers import AutoModel, AutoTokenizer, DebertaV2Model, T5Tokenizer, T5ForConditionalGeneration
from torch.utils import data
from tqdm import tqdm

import sys
sys.path.append("../")
from tasks.coref.ecbp.preprocess import preprocess_ecbplus_coref
from tasks.coref.ontonotes.preprocess import preprocess_ontonotes_coref
from tasks.coref.genia.preprocess import preprocess_genia_coref
from tasks.coref.ecbp.evaluate import eval_ecbplus
from tasks.coref.ontonotes.evaluate import eval_ontonotes
from tasks.coref.ecbp.inference import inference_coref
from utils import restrict_vocab

GPU_ID='1'
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
#device = 'cpu'



class CorefDataset(data.Dataset):
    def __init__(self, prompts, labs):
        self.labs = labs
        self.prompts = prompts

    def __getitem__(self, idx):
        return  self.prompts[idx], self.labs[idx]

    def __len__(self):
        return len(self.labs)



class CorefClassifier(torch.nn.Module):
    def __init__(self, train_file, dev_file, test_file, dataset_name):
        super(CorefClassifier,self).__init__()
        preprocess_dict = {"ecbp": preprocess_ecbplus_coref, "ontonotes": preprocess_ontonotes_coref, "genia": preprocess_genia_coref}
        self.train_df = preprocess_dict[dataset_name](train_file)
        self.dev_df = preprocess_dict[dataset_name](dev_file)
        self.test_df = preprocess_dict[dataset_name](test_file)

        self.tokenizer = T5Tokenizer.from_pretrained("allenai/macaw-3b", add_prefix_space=True) 
        self.model = T5ForConditionalGeneration.from_pretrained("allenai/macaw-3b").to(device)
    

    def process_prompts(self, df):
        prompts = []
        labels = []
        for ix, row in df.iterrows():
            query = f"""$answer$ ; $mcoptions$=(A) Yes (B) No  ; {row["sentence"]} Does {row["entity1"]} refer to {row["entity2"]}? <extra_id_0>"""
            prompts.append(query)
            labels.append(f"""$answer$ = {row["answer"]}""")
        
        return prompts, labels


    def process_eval_prompts(self, df):
        prompts = []
        labels = []
        for ix, row in df.iterrows():
            query = f"""$answer$ ; $mcoptions$=(A) Yes (B) No  ; {row["sentence"]} Does {row["entity1"]} refer to {row["entity2"]}? """
            prompts.append(query)
            labels.append(row["answer"])
        
        return prompts, labels




    def train(self, model_dir, lr = 0.0000001, max_epochs=20, e_stop=5):
        train_prompts, train_labs = self.process_prompts(self.train_df)
        train_dataset = CorefDataset(train_prompts, train_labs)
        train_loader = data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=4)

        dev_prompts, dev_labs = self.process_eval_prompts(self.dev_df)
        dev_dataset = CorefDataset(dev_prompts, dev_labs)
        dev_loader = data.DataLoader(dataset=dev_dataset, shuffle=False, batch_size=2)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        Path(model_dir).mkdir(exist_ok=True, parents=True)

        best_metric = 0
        no_improv = 0

        for ep in range(max_epochs):
            print(f"Epoch {ep+1}")
            tr_loss = []

            self.model.train()

            for pro, labs in tqdm(train_loader):
                input_ids = self.tokenizer(pro, return_tensors="pt", padding=True).input_ids
                lab_ids = self.tokenizer(labs,return_tensors="pt", padding=True).input_ids
                outputs = self.model(input_ids = input_ids.to(device), labels=lab_ids.to(device))
                
                loss = outputs.loss
                tr_loss.append(loss.item())
            
                loss.backward()
                optimizer.step()
                
                
            print(f"Train Loss: {np.mean(tr_loss)}")

            metric = self.evaluate(dev_loader)
            print(f"Dev F1: {metric}")     
            if metric > best_metric:
                no_improv = 0
                best_metric = metric
                torch.save({
                    "model_state_dict" : self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                    }, Path(model_dir, str(ep+1)))
            else:
                no_improv += 1
                if no_improv == e_stop:
                    print(f"Stopping early to avoid over fitting")
                    exit()



    def evaluate(self, eval_loader):
        restriction = ["$answer$ = Yes", "$answer$ = No"]
        max_len = 10
        beam_size = 20

        self.model.eval()

        def restrict_decode_vocab(batch_idx, prefix_beam):
            """ Function to restrict decode vocab to some tokens. Source: https://github.com/huggingface/transformers/issues/15169"""
            return self.tokenizer(restriction, add_special_tokens=True, return_tensors="pt", is_split_into_words=True)['input_ids'].tolist()
        pred_ans = []
        gold_labs = []

        for pro, labs in tqdm(eval_loader):
            input_ids = self.tokenizer(pro, return_tensors="pt", padding=True).input_ids
            lab_ids = self.tokenizer(labs,return_tensors="pt", padding=True).input_ids
            
            with torch.no_grad():
                outputs = self.model.generate(input_ids.to(device), num_return_sequences=beam_size, num_beams=beam_size, prefix_allowed_tokens_fn= restrict_decode_vocab, max_length=max_len, return_dict_in_generate=True)

                gold_labs.extend(labs)

                for ix in range(input_ids.shape[0]):
                    cand_outs = torch.reshape(outputs.sequences, (input_ids.shape[0],beam_size, int((outputs.sequences.shape[0])*(outputs.sequences.shape[1])/(input_ids.shape[0]*beam_size))))
                    for seq_ix in range(beam_size): #Iterate over the beam size
                        output_ans = self.tokenizer.decode(cand_outs[ix][seq_ix], skip_special_tokens=True)
                        if output_ans.strip() == "$answer$ = Yes":
                            pred_ans.append("Yes")
                            break

                        elif output_ans.strip() == "$answer$ = No":
                            pred_ans.append("No")
                            break
                torch.cuda.empty_cache()

        f_score = f1_score(gold_labs , pred_ans, average='macro')  
         
        return f_score








    def predict(self, eval_loader):
        restriction = ["$answer$ = Yes", "$answer$ = No"]
        max_len = 10
        beam_size = 20

        generations = []

        self.model.eval()

        def restrict_decode_vocab(batch_idx, prefix_beam):
            """ Function to restrict decode vocab to some tokens. Source: https://github.com/huggingface/transformers/issues/15169"""
            return self.tokenizer(restriction, add_special_tokens=True, return_tensors="pt", is_split_into_words=True)['input_ids'].tolist()
        
        for pro, labs in tqdm(eval_loader):
            input_ids = self.tokenizer(pro, return_tensors="pt", padding=True).input_ids
            lab_ids = self.tokenizer(labs,return_tensors="pt", padding=True).input_ids
 
            with torch.no_grad():
                outputs = self.model.generate(input_ids.to(device), num_return_sequences=beam_size, num_beams=beam_size, output_scores=True, return_dict_in_generate=True, prefix_allowed_tokens_fn= restrict_decode_vocab, max_length=max_len)
                
                candidate_sequences = torch.reshape(outputs.sequences, (input_ids.shape[0], beam_size, int((outputs.sequences.shape[0])*(outputs.sequences.shape[1])/(input_ids.shape[0]*beam_size))))
                candidate_sequences_scores = torch.reshape(outputs.sequences_scores, (input_ids.shape[0], beam_size))
                
                for in_batch_ix in range(input_ids.shape[0]):
                    prompt_gens = []

                    val_answers = restriction.copy()
                    ans_scores = [0]*len(val_answers)
                    
                    inst_sequences = candidate_sequences[in_batch_ix]
                    inst_sequence_scores = candidate_sequences_scores[in_batch_ix]
    
                    # Process outputs into a list of dictionary to store the generation and scores
                    passed_ans = []
                    for seq_ix in range(beam_size): #Iterate over the beam size
                        seq_dict = {}
                        sentence = self.tokenizer.decode(inst_sequences[seq_ix], skip_special_tokens=True).replace(",", " ,")
                        score = inst_sequence_scores[seq_ix].item()
                    
                        # The outputs are restrictive based on the 'val_answers' list. 
                        # We post-process these task outputs separately
                        # Only consider answers in the valid answer list and ones thta have not been generated yet
                        if (sentence in val_answers) and (sentence not in passed_ans):
                            ans_scores[val_answers.index(sentence)] = score
                            if sentence not in passed_ans:
                                passed_ans.append(sentence)
                        if len(passed_ans) == len(val_answers):
                            break
                
                    # We need to post process the output and save the generations in descending order of score. 
                    # If needed these scores might have to be calibrated 
                    out_sf = torch.Tensor(ans_scores)
                    recalib_scores = out_sf
                    
                    # Sort by descending score
                    temp = sorted(zip(restriction.copy(), recalib_scores.tolist()),key=lambda i:i[1],reverse=True)
                    
                    # Post-process answers to Yes/No
                    for l1, l2 in temp:
                        if l1 == "$answer$ = No":
                            prompt_gens.append({"sentence":"No", "score":l2})
                        elif l1 == "$answer$ = Yes":
                            prompt_gens.append({"sentence":"Yes", "score":l2})
                   
                    generations.append(prompt_gens)
                torch.cuda.empty_cache() 

        return generations


def add_parser_args(parser):
    parser.add_argument('--dataset_name', default= "ecbp", type=str, choices=['ecbp','ontonotes','genia'])
    parser.add_argument('--mode', default="train", type=str, choices=['train','test'])
    parser.add_argument('--read_generated', action='store_true')
    parser.add_argument('--read_inferences', action='store_true')
    return parser



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = add_parser_args(parser)
    args = vars(parser.parse_args())
 

    dataset_name = args["dataset_name"]
    mode = args["mode"]
    run_generate = not args['read_generated']
    run_inferences = not args['read_inferences']

    if dataset_name == "ecbp":
        DATA_DIR =  "../../data/awesomecoref/processed_ecb/data/ecb/gold_singletons/" 
        train_file = DATA_DIR + "train_entities_corpus_level.conll"
        dev_file   = DATA_DIR + "dev_entities_corpus_level.conll"
        test_file = DATA_DIR + "test_entities_corpus_level.conll"
    elif dataset_name == "ontonotes":
        DATA_DIR =  "../../data/conll-2012/v12/data/{}/data/english/annotations/" 
        train_file = DATA_DIR.format("train")
        dev_file = DATA_DIR.format("development")
        test_file = DATA_DIR.format("test")  
    elif dataset_name == "genia":
        DATA_DIR =  "../../data/GENIA_MedCo_coreference_corpus_1.0/{}" 
        train_file = DATA_DIR.format("train")
        dev_file = DATA_DIR.format("dev")
        test_file = DATA_DIR.format("test")  



    model_dir = f"../../models/sup_baseline/coref/{dataset_name}/{SEED}/"

    coref_model = CorefClassifier(train_file, dev_file, test_file, dataset_name)
    if mode == "train":
        coref_model.train(model_dir=model_dir)
    else:
        split = "test"
        if split == "dev":
            eval_df = coref_model.dev_df
        elif split == "test":
            eval_df = coref_model.test_df
        
        model_path = model_dir + "1"
        
        eval_prompts, eval_labs = coref_model.process_eval_prompts(eval_df)
        eval_dataset = CorefDataset(eval_prompts, eval_labs)
        eval_loader = data.DataLoader(dataset=eval_dataset, shuffle=False, batch_size=2)

        checkpoint = torch.load(model_path)
        adapted_dict = {"model."+k:v for k,v in checkpoint["model_state_dict"].items()}
        coref_model.load_state_dict(adapted_dict)

        if run_generate:
            generations = coref_model.predict(eval_loader)
            with open(f"./dumps/{dataset_name}_coref_{SEED}_gens.bin", "wb") as out:
                pickle.dump(generations, out)
        else:
            with open(f"./dumps/{dataset_name}_coref_{SEED}_gens.bin", "rb") as out:
                generations = pickle.load(out)



        eval_dict = {"ecbp": eval_ecbplus, "ontonotes": eval_ontonotes, "genia": eval_ontonotes}

        # Unconstrained 
        uncon_gens = [gen[0]["sentence"] for gen in generations]
        meta = {"gold_dump_file": f"./../../results/coref/{dataset_name}_finetuned_{SEED}_gold.txt", "pred_dump_file": f"./../../results/coref/{dataset_name}_finetuned_{SEED}_uncons.txt", "constrained":False}
        eval_dict[dataset_name](eval_df, uncon_gens, meta)


        class dotdict(dict):
            """dot.notation access to dictionary attributes"""
            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__
        
        # Constrained
        infer_meta = {"thresh": 0.5, "config": dotdict({"score_type":"raw"})}
        
       
        if run_inferences:
            cons_ans = inference_coref(eval_df, generations, sanity_check=False, meta=infer_meta)
            with open(f"./dumps/{dataset_name}_coref_{SEED}_consans.bin", "wb") as out:
                pickle.dump(cons_ans, out)
        else:
            with open(f"./dumps/{dataset_name}_coref_{SEED}_consans.bin", "rb") as out:
                cons_ans = pickle.load(out)
         
        
        meta = {"gold_dump_file": f"./../../results/coref/{dataset_name}_finetuned_{SEED}_gold.txt", "pred_dump_file": f"./../../results/coref/{dataset_name}_finetuned_{SEED}_cons.txt", "constrained":True}
        eval_dict[dataset_name](eval_df, cons_ans, meta)






