from configparser import ConfigParser

import itertools
import numpy as np
import spacy
from typing import Union, List, Tuple
import uuid

nlp = spacy.load("en_core_web_sm")

class Config():
    def __init__(self, filename="config.ini"):
        """ Constructor for the Config class
        Inputs
        -----------------
        filename: str or pathlib.Path. Config file path
        """
        self.config_file = filename
        self.read_config(self.config_file)


    def read_config(self, filename):
        """Reads data from the config file into a config class
        """
        config = ConfigParser()
        config.readfp(open(filename))

        # Read details mentioned in the config file
        # to the class object
        self.task_name = config.get('Meta', 'task_name')
        self.dataset_name = config.get('Meta','dataset_name')
        self.data_dir = config.get('Data','data_dir')
        self.data_file = config.get('Data','data_file')
        self.few_shot_file = config.get('Data', 'few_shot_file')
        self.mode = config.get('Run','mode')
        self.few_shot = self.check_bool(config.get('Run','few_shot'))

        self.num_shots = int(config.get('Few Shot', 'num_shots'))
        self.shot_type = config.get('Few Shot', 'shot_type')
        self.shot_seed = int(config.get('Few Shot', 'shot_seed'))
        self.order_num = int(config.get('Few Shot', 'order_num'))

        self.model = config.get('Model','model')
        self.prompt_type = config.get('Prompt','prompt_type')
        self.prompt_style = config.get('Prompt','prompt_style')
        self.context_style = config.get('Prompt', 'context_style')
        self.read_spec = config.get('Dumps', 'read_spec')
        self.spec_det = config.get('Dumps', 'dump_spec')
        
        self.do_calibrate = self.check_bool(config.get('Calibration', 'do_calibrate'))
        
        self.calibration_type = config.get('Calibration','calibration_type')
        self.score_type = config.get('Inference','score_type')

        self.consistency_check()

    def check_bool(self, spec):
        if spec == "True":
            return True
        else:
            return False


    def consistency_check(self):
        if self.calibration_type == "calib_before_use" and self.score_type!="prob":
            raise Exception(f"""Calibration method {self.calibration_type} is not compatible with the score type {self.score_type}. Please change the score type to "prob" in your config file""")
        if self.calibration_type == "score_diff" and self.score_type!="raw":
            raise Exception(f"""Calibration method {self.calibration_type} is not compatible with the score type {self.score_type}. Please change the score type to "raw" in your config file""")






def restrict_vocab(config: Config) -> Tuple[ Union[List[str], None], Union[int, None], Union[str, None]]:
    """ Produce output vocab restrictions based on the 
    task and model.
    Inputs
    -------------------------
    config: Config. The config instance

    Outputs
    ------------------------
    restriction: List[str]. Contains list of tokens which the generatot should be restricted to.
                    E.g.: ["Yes","No"] for a Yes/No task.
    max_len: int. Max number of tokens which can be generated.
    calib_prompt: str. Dummy prompt used for calibration
    """
    restriction = None # Contains restrictions on what can be generated
    max_len = None # Maximum length in terms of sub-words used for generation
    calib_prompt = None # Dummy prompt for calibration

    if config.task_name in ["coref"]:
        if config.model in ["t5", "t5-11b", "t5-3b"]:
            restriction = ["Yes","No"]
            max_len = 3
            calib_prompt = "question: Yes or No? context: "
        elif config.model in ["macaw-3b", "macaw-large"]:
            restriction = ["$answer$ = Yes", "$answer$ = No"]   # Restriction on vocabulary
            max_len = 10
            calib_prompt = "$answer$ ; $mcoptions$= (A) Yes (B) No ; Yes or No?"
        else:
            raise "Model not in supported list of models. Please add it in utils.py or use a pre-existing one"
    return restriction, max_len, calib_prompt




def fetch_root(txt):
    """ Returns root of the phrase/sent.
    """
    for tok in  nlp(txt):
            if tok.dep_ == "ROOT":
                return tok.text




def analyse_beams(targets, generations, root_analysis=False):
    """ Analysing the prevelance of targets in the generated 
    sequences. Setting root_analysis to True performs a root
    match when targets are phrases (and not words).
    """
    absent = 0
    beam_ranks = []
    if root_analysis:
        absent_rt = 0
        beam_ranks_rt = []

    
    for ix, gen in enumerate(generations):
        target_string = targets[ix]
        found = False

        for beam_ix in range(1,len(gen)+1):
            if target_string == gen[beam_ix-1]["sentence"]:
                found = True    # Match Found
                beam_ranks.append(beam_ix)  # What is the position of the sequence
                break
        # No match found
        if not found:
            absent += 1

        ## Sub module to compute statitics based on the roots
        # This can be helpful for tasks like SRL
        if root_analysis:
            target_rt = fetch_root(targets[ix])
            found = False

            for beam_ix in range(1,len(gen)+1):
                seq_rt = fetch_root(gen[beam_ix-1]["sentence"])   # Fetch root of the sequence
                if target_rt == seq_rt:
                    found = True
                    beam_ranks_rt.append(beam_ix)
                    break

            if not found:
                absent_rt += 1
    
    print(f"Ratio of target sequences outside beams: {absent/len(generations)}")
    print(f"Average position of target seq in beam: {np.mean(beam_ranks)}")
    if root_analysis:
        print(f"Ratio of target roots outside beams: {absent_rt/len(generations)}")
        print(f"Average position of target root in beam: {np.mean(beam_ranks_rt)}")





class ValDict():
    """ Value Dictionary which maps a unique identifier 
    to an entity instance and vice versa.
    """
    def __init__():
        self.dict = {}

    def get_key(self, v):
        for k in self.dict.keys():
            if self.dict[k] == v:
                return k

    def forward(self, val):
        if val not in self.dict.values():
            key = uuid.uuid4()
            self.dict[key] = val
            return key
        else:
            return self.get_key(val)
            
        

def get_highlighted_context(row, model, full_context=False):
    if model in ['t5','t5-11b','t5-3b','unified-qa','macaw-3b','macaw-large']:
        
        s1 = row['sent1'].copy()
        ent1_ix = row['ent1_ix']
        s1[ent1_ix[0]] = "*"+s1[ent1_ix[0]]
        s1[ent1_ix[-1]] = s1[ent1_ix[-1]]+"*"
        
        if full_context:
            passage = row['passage'].copy()
            passage[row['sent1_id']] = s1

        if row['sent1_id'] == row['sent2_id']:
            s2 = s1.copy()    
        else:
            s2 = row['sent2'].copy()
        ent2_ix = row["ent2_ix"]
        s2[ent2_ix[0]] = "*"+s2[ent2_ix[0]]
        s2[ent2_ix[-1]] = s2[ent2_ix[-1]]+"*"
        

        if full_context:
            passage[row['sent2_id']] = s2
            context = " ".join(list(itertools.chain(*passage)))
        else:
            if row["sent1_id"] == row['sent2_id']:
                context = " ".join(s2)
            else:
                if row['in_order']:
                    context =  " ".join(s1) + " " + " ".join(s2)
                else:
                    context =  " ".join(s2) + " " + " ".join(s1)
        
    return context





def dataset_document_iterator(file_path):
    """ Reads data from CONLL formated files and extarcts documents
    Source from https://huggingface.co/datasets/conll2012_ontonotesv5/blob/main/conll2012_ontonotesv5.py
    """
    documents = []
    with open(file_path, "r", encoding="utf8") as open_file:
        conll_rows = []
        document  = []
        for line in open_file:
            line = line.strip()
            if line != "" and not line.startswith('#'):
                conll_rows.append(line)
            else:
                if conll_rows:
                    document.append(conll_rows)
                    conll_rows = []
            if line.startswith("#end document"):
                documents.append(document)
                document = []
        if document:
            documents.append(document) 
        
    return documents




def right_to_left_search(rel_ids, max_mentions):
    """ Anaphora resolution heuristic
    """
    #prinre
    rel_mat = np.full((max_mentions,max_mentions),"N", dtype=str)
    for rel in rel_ids:
        low = min(rel)
        high = max(rel)
        rel_mat[low][high] = "Y"
    
    clusters = [[0]]
    viol = 0
    transitivity_viol = 0

    # Computing transivity violations
    for i in range(1, max_mentions):
        for j in range(i+1, max_mentions):
            for k in range(j+1, max_mentions):
                trans_sum = 0
                if rel_mat[i][j] == "Y":
                    trans_sum += 1
                if rel_mat[j][k] == "Y":
                    trans_sum += 1
                if rel_mat[i][k] == "Y":
                    trans_sum += 1
                
                # The transitivity constrint is broken if exactly two 
                # edges exists between the mentions
                if trans_sum == 2:
                    transitivity_viol += 1


    for i in range(1,max_mentions):
        flag = True
        cluster_id = None
        for j in range(i-1,-1,-1):
            # Analyse if the model says coreferrent
            if rel_mat[j][i] == "Y":
                if flag:
                    # Condition when the nearest antexedent matches
                    for c_ix, clus in enumerate(clusters):
                        if j in clus:
                            clusters[c_ix].append(i)
                            flag = False
                            cluster_id = c_ix
                            break
                else:
                    # All other antecedents not in the same 
                    # cluster are considered violations
                    if j not in clusters[cluster_id]:
                        viol += 1
        if flag:
            clusters.append([i])
        
       
    return clusters, transitivity_viol



def get_modified_ans(clusters, all_relations):
    # In case the where there were violations and corrections
    # were to be made, we need to get the modified answers for the prompts
    
    modified_ans = []
    for rel in all_relations:
        ent1_clus = None
        ent2_clus = None
        for c_ix, clus in enumerate(clusters):
            if rel[0] in clus:
                ent1_clus = c_ix 
                break
        for c_ix, clus in enumerate(clusters):
            if rel[1] in clus:
                ent2_clus = c_ix 
                break

        assert ent1_clus != None
        assert ent2_clus != None

        if ent1_clus == ent2_clus:
            modified_ans.append("Yes")
        else:
            modified_ans.append("No")
    
    return modified_ans.copy() 






def check_violations(yes_ids, no_ids, max_mentions):
    rel_mat = np.full((max_mentions,max_mentions),"N", dtype=str)
    for rel in yes_ids:
        low = min(rel)
        high = max(rel)
        rel_mat[low][high] = "Y"
   
    for rel in no_ids:
        low = min(rel)
        high = max(rel)
        rel_mat[low][high] = "N"
    
    
    transitivity_viol = 0
    total_checks = 0

    # Computing transivity violations
    for i in range(1, max_mentions):
        for j in range(i+1, max_mentions):
            for k in range(j+1, max_mentions):
                at_least_two_edges = False 
                if (rel_mat[i][j] == "Y") and (rel_mat[j][k] == "Y"):
                    if rel_mat[i][k] != "N/A":
                        at_least_two_edges = True
                    if rel_mat[i][k] == "N":
                        transitivity_viol += 1
                        #print(i,j,k)
                elif (rel_mat[i][k] == "Y") and (rel_mat[j][k] == "Y"):
                    if rel_mat[i][j] != "N/A":
                        at_least_two_edges = True
                    if rel_mat[i][j] == "N":
                        transitivity_viol += 1
                        #print(i,j,k)
                elif (rel_mat[i][j] == "Y") and (rel_mat[i][k] == "Y"):
                    if rel_mat[j][k] != "N/A":
                        at_least_two_edges = True
                    if rel_mat[j][k] == "N":
                        transitivity_viol += 1
                        #print(i,j,k)

                if at_least_two_edges:
                    total_checks += 1
    
    return transitivity_viol, total_checks



if __name__ == "__main__":
    conf = Config()
    print(conf.task_name)
    print(conf.data_file)

