import argparse
import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import f1_score
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils import data
from tqdm import tqdm

import sys
sys.path.append("../")
from tasks.srl.wikisrl.preprocess import preprocess_wikisrl
from tasks.srl.qasrl2.preprocess import preprocess_qasrl2
from tasks.srl.wikisrl.inference import inference_srl
from tasks.srl.wikisrl.evaluate import eval_wikisrl


GPU_ID='1'
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'



class SRLDataset(data.Dataset):
    def __init__(self, prompts, labs):
        self.labs = labs
        self.prompts = prompts

    def __getitem__(self, idx):
        return  self.prompts[idx], self.labs[idx]

    def __len__(self):
        return len(self.labs)



class SRLExtractor(torch.nn.Module):
    def __init__(self, train_file, dev_file, test_file, dataset_name):
        super(SRLExtractor,self).__init__()
        preprocess_dict = {"wiki": preprocess_wikisrl, "qasrl2": preprocess_qasrl2}
        self.train_df = preprocess_dict[dataset_name](train_file)
        self.dev_df = preprocess_dict[dataset_name](dev_file)
        self.test_df = preprocess_dict[dataset_name](test_file)
        self.dataset_name = dataset_name
        
        self.tokenizer = T5Tokenizer.from_pretrained("t5-3b") 
        self.model = T5ForConditionalGeneration.from_pretrained("t5-3b").to(device)
    

    def process_prompts(self, df):
        prompts = []
        labels = []
        for ix, row in df.iterrows():
            if self.dataset_name == "wiki":
                query = f"""question: {row['question']} context: {row['sentence']} <extra_id_0>"""
            elif self.dataset_name == "qasrl2":
                query = f"""question: {row['question']} context: {" ".join(row['sentence'])} <extra_id_0>"""
 
            prompts.append(query)
            ans = row["answer"].split(" ### ")[0]
            labels.append(f"""<extra_id_0> {ans}""")
        
        return prompts, labels


    def process_eval_prompts(self, df):
        prompts = []
        labels = []
        for ix, row in df.iterrows():
            if self.dataset_name == "wiki":
                query = f"""question: {row['question']} context: {row['sentence']}"""
            elif self.dataset_name == "qasrl2":
                query = f"""question: {row['question']} context: {" ".join(row['sentence'])}"""
 
            prompts.append(query)
            labels.append(row["answer"])
        
        return prompts, labels


    def train(self, model_dir, lr = 0.000001, max_epochs=20, e_stop = 5):
        train_prompts, train_labs = self.process_prompts(self.train_df)
        train_dataset = SRLDataset(train_prompts, train_labs)
        train_loader = data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=8)

        dev_prompts, dev_labs = self.process_eval_prompts(self.dev_df)
        dev_dataset = SRLDataset(dev_prompts, dev_labs)
        dev_loader = data.DataLoader(dataset=dev_dataset, shuffle=False, batch_size=4)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        Path(model_dir).mkdir(exist_ok=True, parents=True)

        best_metric = 0
        no_improv = 0

        for ep in range(max_epochs):
            print(f"Epoch {ep+1}")
            tr_loss = []

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
            print(f"Dev exact accuracy: {metric}")
        
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
        pred_ans = []
        gold_labs = []

        for pro, labs in tqdm(eval_loader):
            input_ids = self.tokenizer(pro, return_tensors="pt", padding=True).input_ids
            lab_ids = self.tokenizer(labs,return_tensors="pt", padding=True).input_ids
            
            with torch.no_grad():
                outputs = self.model.generate(input_ids.to(device), num_return_sequences=1, num_beams=20)
                gold_labs.extend(labs)
            
                for ix in range(outputs.shape[0]):
                    output_ans = self.tokenizer.decode(outputs[ix], skip_special_tokens=True)
                    pred_ans.append(output_ans.strip())
            torch.cuda.empty_cache() 
        corr = 0                    
        for ix in range(len(pred_ans)):
            if pred_ans[ix] in gold_labs[ix].split(" ### "):
                corr += 1

        return corr/ len(pred_ans)



    def predict(self, eval_loader):
        beam_size = 20
        generations = []
       
        for pro, labs in tqdm(eval_loader):
            input_ids = self.tokenizer(pro, return_tensors="pt", padding=True).input_ids
             
            with torch.no_grad():
                outputs = self.model.generate(input_ids.to(device), num_return_sequences=beam_size, num_beams=beam_size, output_scores=True, return_dict_in_generate=True)
                
                candidate_sequences = torch.reshape(outputs.sequences, (input_ids.shape[0], beam_size, int((outputs.sequences.shape[0])*(outputs.sequences.shape[1])/(input_ids.shape[0]*beam_size))))
                candidate_sequences_scores = torch.reshape(outputs.sequences_scores, (input_ids.shape[0], beam_size))
                
                for in_batch_ix in range(input_ids.shape[0]):
                    prompt_gens = []
 
                    inst_sequences = candidate_sequences[in_batch_ix]
                    inst_sequence_scores = candidate_sequences_scores[in_batch_ix]
    
                    # Process outputs into a list of dictionary to store the generation and scores
                    passed_ans = []
                    for seq_ix in range(beam_size): #Iterate over the beam size
                        seq_dict = {}
                        sentence = self.tokenizer.decode(inst_sequences[seq_ix], skip_special_tokens=True).replace(",", " ,")
                        score = inst_sequence_scores[seq_ix].item()
                        
                        prompt_gens.append({"sentence": sentence, "score":score})
                                          
                    generations.append(prompt_gens)

        return generations




def add_parser_args(parser):
    parser.add_argument('--dataset_name', default= "wiki", type=str, choice=['wiki','qasrl2'])
    parser.add_argument('--mode', default="train", type=str, choices=['train','test'])
    parser.add_argument('--best_model', default = 1, type=int)
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
    
    DATA_DIR =  "../../data/"

    if dataset_name == "wiki": 
        train_file = DATA_DIR + "wiki1.train.qa"
        dev_file = DATA_DIR + "wiki1.dev.qa"
        test_file = DATA_DIR + "wiki1.test.qa"
    elif dataset_name == "qasrl2":
        train_file = DATA_DIR + "qasrl-v2/orig/train.jsonl"
        dev_file   = DATA_DIR + "qasrl-v2/orig/dev.jsonl"
        test_file = DATA_DIR + "qasrl-v2/orig/test.jsonl"

    model_dir = f"../../models/sup_baseline/srl/{dataset_name}/{SEED}/"

    srl_model = SRLExtractor(train_file, dev_file, test_file, dataset_name)
    if mode == "train":
        srl_model.train(model_dir=model_dir)
    else:
        split = "test"
        if split == "dev":
            eval_df = srl_model.dev_df
        elif split == "test":
            eval_df = srl_model.test_df
        
        model_path = model_dir + str(args['best_model'])
        
        eval_prompts, eval_labs = srl_model.process_eval_prompts(eval_df)
        eval_dataset = SRLDataset(eval_prompts, eval_labs)
        eval_loader = data.DataLoader(dataset=eval_dataset, shuffle=False, batch_size=2)

        checkpoint = torch.load(model_path)
        coref_model.load_state_dict(checkpoint["model_state_dict"])

        if run_generate:
            generations = srl_model.predict(eval_loader)
            with open(f"./dumps/{dataset_name}_srl_{SEED}_gens.bin", "wb") as out:
                pickle.dump(generations, out)
        else:
            with open(f"./dumps/{dataset_name}_srl_{SEED}_gens.bin", "rb") as out:
                generations = pickle.load(out)
        
        uncon_gens = [gen[0]["sentence"] for gen in generations]
            
        # Unconstrained 
        meta = {"gold_dump_file": f"./../../results/coref/{dataset_name}_finetuned_{SEED}_gold.txt", "pred_dump_file": f"./../../results/coref/{dataset_name}_finetuned_{SEED}_uncons.txt", "constrained":False}
        eval_wikisrl(eval_df, uncon_gens, meta)

        # Constrained
        infer_meta = {"thershold": 0.5, "config": {"score_type":"raw"}}
        if run_inferences:
            cons_ans = inference_srl(eval_df, generations, sanity_check=False, meta=infer_meta)
            with open(f"./dumps/{dataset_name}_srl_{SEED}_consans.bin", "wb") as out:
                pickle.dump(cons_ans, out)
        else:
            with open(f"./dumps/{dataset_name}_srl_{SEED}_consans.bin", "rb") as out:
                cons_ans = pickle.load(out)
            
 
        meta = {"gold_dump_file": f"./../../results/coref/{dataset_name}_deberta_{SEED}_gold.txt", "pred_dump_file": f"./../../results/coref/{dataset_name}_deberta_{SEED}_cons.txt", "constrained":True}
        eval_wikisrl(eval_df, cons_ans, meta)






