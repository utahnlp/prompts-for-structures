#from bs4 import BeautifulSoup
import json
import os
import pandas as pd
from pathlib import Path
import re
from tqdm import tqdm
import xmltodict
import xml.etree.ElementTree as ET

from tasks.coref.ecbp.preprocess import generate_examples

class XMLParser():
    def __init__(self, filename):
        with open(filename) as fd:
            self.lines = fd.readlines()
    
    def get_docid(self):
        """ Gets article ID from the lines 
        """
        for line in self.lines:
            if line[:6] == "<PMID>":
                p = re.compile("<PMID>(.*)</PMID>")
                result = p.findall(line)
                return result[0]
        raise Exception("No <PMID> found")

    def get_sent_lines(self):
        self.sent_ids = []
        self.sent_lines = []

        for l_ix, line in enumerate(self.lines):
            if line[:9] == "<sentence":
                p = re.compile("""<sentence id="S(..?)">""")
                result = p.findall(line)
                self.sent_ids.append(int(result[0])-1)
                self.sent_lines.append(l_ix)



    def process_coref_data(self):
        coref_dict = {"sentences": [], "entities": {}, "sent_ixs": []}
        men_id = 0
        ent_dict = {}
        for l_ix, line_id in enumerate(self.sent_lines):
            coref_dict["sent_ixs"].append(0)
            sent_cnt_ix = l_ix

            line = self.lines[line_id]
            sent_p = re.compile("""<sentence id="S(..?)">(.*)</sentence>""")
            line_data = sent_p.findall(line)[0][1]

            token_list = []
            coref_open = False
            meta_open = False
            token_id = -1
            ent_id = None
            skip = 0

            line_data = line_data.replace(" <","<")
            line_data = line_data.replace("<", " <")
            line_data = line_data.replace("> ",">")
            line_data = line_data.replace(">", "> ")

            for token in line_data.split():
                if meta_open:
                    if token[:3] == "ref":
                        try:
                            tok_p = re.compile("ref=\"(.*)\"")
                            ent_id = tok_p.findall(token)[0]
                        except IndexError:
                            tok_p = re.compile("ref=\"(.*)")
                            ent_id = tok_p.findall(token)[0]
                            
                            

                    if token[:2] == "id":
                        tok_p = re.compile("id=\"(.*)\"")
                        try:
                            inter_ent = tok_p.findall(token)[0].replace(">","")
                        except IndexError:
                            print(line_data)
                            print(token)
                            exit()

                        if ent_id == None:
                            ent_id = inter_ent
                        else:
                            parent = ent_id
                            while True:
                                if ent_id in ent_dict.keys():
                                    ent_id = ent_dict[ent_id]
                                else:
                                    break
                            ent_dict[inter_ent] = ent_id
                        
                        meta_open = False
                    continue

                if token == "<coref":
                    if coref_open:
                        skip += 1
                        continue
                    # Case where coref 
                    coref_open = True
                    meta_open = True
                    tok_idx = []
                elif "</coref>" in token:
                    # Closing Multi-word reference text
                    if skip !=0:
                        # Ignore if nested entities
                        skip -= 1
                        continue
                    #token_id += 1
                    #token_list.append(token.replace("</coref>",""))
                    #tok_idx.append(token_id)
                    if ent_id not in coref_dict["entities"].keys():
                        coref_dict["entities"][ent_id] = []    
                    coref_dict["entities"][ent_id].append({"sent_id":sent_cnt_ix, "tok_idx":tok_idx, "mention_id":men_id})
                    men_id += 1
                    coref_open = False
                    ent_id = None
                else:
                    # Normal token
                    if ("id=" in token) or \
                        ("min=" in token) or \
                        ("type=" in token) or \
                        ("ref=" in token):
                        continue
                    token_id += 1
                    token_list.append(token)
                    if coref_open:
                        tok_idx.append(token_id)

            coref_dict["sentences"].append(token_list) 
            #print(line_data.split())
                       
            assert skip == 0 #There should be no open mention
            assert coref_open == False
            assert meta_open == False
        
 
        return coref_dict




def preprocess_doc(filename):
    parser = XMLParser(filename)
    doc_id = parser.get_docid()
    
    parser.get_sent_lines()
    coref_dict = parser.process_coref_data() 
    
    examples = generate_examples(coref_dict, doc_id)

    return examples
    #doc_id = data.find("PMID").text
    #article = data.find("Article")

    # Process the title
    #for sent in article.iter('sentence'):
    #    print(list(sent.itertext()))
    #    terms = [elem.text for elem in sent.iter('coref') if elem.text]
    #    print(terms)
    #    print()
        #sent = [''.join(elem.itertext()) for elem in article.iter('sentence')]
        #print(sent)
    #article_title = article.find("ArticleTitle")
    #sentence = article_title.find("sentence")
    #for child in sentence:
    #    print(child.tag, child.text)
    #for text in sentence.text:
    #   print(text)
    #print(len(article.getchildren()))
    #print(len(article_title.getchildren()))
    #print(len(sentence.getchildren()))
    #title = d


def preprocess_genia_coref(filepath):
    """ Preprocess files to dataframe for the Genia dataset
    Input
    ------------
    filepath - str or Path. Folder path to the XML files

    Output
    -----------
    df - pd.DataFrame. Dataframe containing the processed data.
    """
    files = list(Path(filepath).glob("*.xml"))
    files.sort()
    
    doc = []
    # Iterating over all the files in the set
    for f in tqdm(files):
        if str(f)[-12:] in ["10214854.xml","/8513868.xml"]: #Skipping
            continue
        
        examples  = preprocess_doc(f)
        doc.extend(examples)
    columns = ["doc_id","passage","sentence","answer","entity1","entity2","entity1_id","entity2_id","sent1","sent2","sent1_id","sent2_id","ent1_ix","ent2_ix","in_order","ent1_ix_glob", "ent2_ix_glob", "mention_id1", "mention_id2"]
    data_df = pd.DataFrame(doc, columns=columns)

    return data_df
