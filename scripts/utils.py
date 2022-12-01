from configparser import ConfigParser

import numpy as np
import spacy
import uuid

nlp = spacy.load("en_core_web_sm")

class Config():
    def __init__(self, filename="config.ini"):
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
        self.mode = config.get('Run','mode')
        self.model = config.get('Model','model')
        self.prompt_type = config.get('Prompt','prompt_type')


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
            
        
    




if __name__ == "__main__":
    conf = Config()
    print(conf.task_name)
    print(conf.data_file)

