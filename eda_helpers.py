from collections import Counter
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from typing import List
import re
import pandas as pd


SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


def sanity_check_entities(lis:List[list])->bool:
    """
    I-TAG_NAME must come only and only after B-TAG_NAME or I-TAG_NAME
    First case where it fails will give the False
    """
    prev = "O"
    for ent in lis:
        if ent.startswith("I"):
            if (prev == "O"): return False
            if prev != ent:
                if prev != "B"+ent[1:]: return False
        prev = ent
    
    return True


def build_parent_level_dict(sub_level_dict:dict, kind = "extend"):
    result = {}

    for key,value in sub_level_dict.items():
        new_key = key.replace("B-","").replace("I-","")
        
        if isinstance(value, (tuple,list)) or kind == "extend":
            if new_key not in result: result[new_key] = value
            else:result[new_key].extend(value)
        
        elif isinstance(value, (str,float)) or kind == "append":
            if new_key not in result: result[new_key] = [value]
            else:result[new_key].append(value)
        
        elif isinstance(value, int) or kind == "add":
            if new_key not in result: result[new_key] = value
            else:result[new_key] += value
        
    return result



def load_data(file_path:str, split:bool = True) -> tuple:
    with open(file_path,"r") as f: raw_data = [x.strip().split("\t") for x in f.readlines()]
    
    tweets_list = []
    entities_list = []

    temp_ent = []
    temp_words = []

    for index, lis in enumerate(raw_data): # lis is a list of [token, entity]
        if lis == [""]:
            assert len(temp_words) == len(temp_ent), "Sanity Check: Irregular Length"
            
            tweets_list.append(temp_words)
            entities_list.append(temp_ent)

            temp_words = []
            temp_ent = []
        else:
            (word,entity) = lis
            word = word.strip()
            entity = entity.strip()

            temp_words.append(word)
            temp_ent.append(entity)


    assert len(tweets_list) == len(entities_list), "entity text length mismatch"
    
    if split:
        train_tokens, val_tokens, train_ent, val_ent = train_test_split(tweets_list, entities_list, test_size=0.2, random_state = SEED)

        len(train_tokens) == len(train_ent), "Sanity Check failed"
        for i in range(len(train_tokens)):
            assert len(train_tokens[i]) == len(train_ent[i]), "Sanity Check Failed"
    
        return train_tokens, val_tokens, train_ent, val_ent

    else: return tweets_list, entities_list
        
        

class EDA:

    def compute_sentence_eda(self, sentences:List[list], entities:List[list]):
        """
        Calculate Sentence based EDA
        args:
            sentences: List of Lost of sentence tokens
            entities: List of list of corresponding entities

        Attributes Computed:
            sent_ent_count: Sentence index vs the count of each entity in it
            sent_tok_count: No of tokens in each Sentence
            sent_len_count: Length of each sentence without space
            sent_upper_count: No of Upper case tokens
            sent_lower_count: No of Lowercase tokens
            sent_title_count: No of Title Case tokens
            sent_number_count: No of "digits" in sentence
            sent_alpha_count: No of pure alphabet tokens
            sent_non_alphanum_count: No of Non-alpha Numeric tokens
            sent_non_ascii_count: No of non-ascii tokens in each sentence
            sent_hashtag_count: No of Hashtags
            sent_link_count: No of links
            contractions_count: No of contractions
            mentions_count: No of mentions @something
        """
        self.sent_ent_count = {} # sentence index vs the count of each entity in it
        self.sent_tok_count = {} # No of tokens  / Words
        self.sent_len_count = {} # Length -> without space
        self.sent_upper_count = {}
        self.sent_title_count = {}
        self.sent_lower_count = {}
        self.sent_number_count = {}
        self.sent_alpha_count = {}
        self.sent_non_alphanum_count = {}
        self.sent_non_ascii_count = {}
        self.sent_hashtag_count = {}
        self.sent_link_count = {}
        self.contractions_count = {}
        self.mentions_count = {}

        self.contractions = [] # For other analysis
        self.mentions = []
        self.hashtags_s = [] # another version with words
        self.sent_ent_dist = {}

        for index, sentence in enumerate(sentences):
            ent_lis = entities[index]

            self.sent_ent_dist[index] = Counter(ent_lis) # distribution of entities in each sentence

            self.sent_ent_count[index] = sum([1 for ent in ent_lis if ent != "O"])
            self.sent_tok_count[index] = len(sentence)
            self.sent_len_count[index] = sum([len(token) for token in sentence])

            self.sent_len_count[index] = sum([len(token) for token in sentence])
            self.sent_len_count[index] = sum([len(token) for token in sentence])
            self.sent_len_count[index] = sum([len(token) for token in sentence])
            self.sent_len_count[index] = sum([len(token) for token in sentence])

            self.sent_upper_count[index] = sum([1 for token in sentence if token.isupper()])
            self.sent_title_count[index] = sum([1 for token in sentence if token.istitle()])
            self.sent_lower_count[index] = sum([1 for token in sentence if token.islower()])
            
            self.sent_number_count[index] = sum([1 for token in sentence if token.isnumeric()])
            self.sent_alpha_count[index] = sum([1 for token in sentence if token.isalpha()])
            self.sent_non_alphanum_count[index] = sum([1 for token in sentence if not token.isalnum()])
            self.sent_non_ascii_count[index] = sum([1 for token in sentence if not token.isascii()])
            
            hash = re.findall("#\s*[a-zA-Z0-9!$%&'@~]+", " ".join(sentence))
            if hash: self.hashtags_s.extend(hash)
            self.sent_hashtag_count[index] = len(hash)
            
            self.sent_link_count[index] = sum([1 for token in sentence if token.startswith("http")])

            cont = re.findall("\w+\s*'\s*\w+", " ".join(sentence))
            if cont: self.contractions.extend(cont)
            self.contractions_count[index] = len(cont)

            ment = re.findall("@\s*\w+", " ".join(sentence))
            if ment: self.mentions.extend(ment)
            self.mentions_count[index] = len(ment)

        
        return {"sent_ent_count": [self.sent_ent_count, "No of non 'O' entities"],
                "sent_tok_count": [self.sent_tok_count, "No of tokens in each Sentence"],
                "sent_len_count": [self.sent_len_count, "Length of each sentence without space"],
                "sent_upper_count": [self.sent_upper_count, "No of UPPER OR Abbreviations tokens"],
                "sent_lower_count": [self.sent_lower_count,"No of lowercase tokens"],
                "sent_title_count": [self.sent_title_count,"No of Title Case tokens"],
                "sent_number_count": [self.sent_number_count,"No of digit tokens in sentence"],
                "sent_alpha_count": [self.sent_alpha_count,"No of pure alphabet tokens"],
                "sent_non_alphanum_count": [self.sent_non_alphanum_count,"No of Non-alpha Numeric tokens"],
                "sent_non_ascii_count": [self.sent_non_ascii_count,"No of non-ascii tokens in each sentence"],
                "sent_hashtag_count": [self.sent_hashtag_count, "No of Hashtags"],
                "sent_link_count": [self.sent_link_count,"Non of links"],
                "contractions_count": [self.contractions_count, "No of contractions"],
                "mentions_count": [self.mentions_count, "No of mentions @something"],
                }
        

    def compute_word_entity_eda(self, sentences_list:List[list], entities_list:List[list]):
        """
        Calculate Word and Entity Level Granular EDA
        args:
            sentences: List of Lost of sentence tokens
            entities: List of list of corresponding entities
        
        Attributes Computed:
            counter: Words and their count
            unique_words: set of unique words
            unique_entities: Count of each entity in the corpus
            entity_len: Length Distribution per entity
            
            title: List of title words per entity 
            lower: List of lowercase words per entity 
            abbrev: List of Abbrevation or TITLE per entity 
            nums: List of numbers per entity 
            alpha: List of pure words per entity 
            not_alnum: List of non Alpha numerical words per entity 
            non_ascii: List of non Ascii words per entity 
            ambiguity: List of words if same word is used in different entities
            links: List of links per entity 
            hashtags: List of links per entity
        """
        self.word_counter = {} # Word Count
        self.unique_words = set()
        self.unique_entities = {}
        self.ambiguous_words = {} # If same word is used in different entities
        self.entity_len = {}
        
        self.title = {}
        self.lower = {}
        self.abbrev_upper = {} # Abbrevation or TITLE
        self.nums = {}
        self.alpha = {}
        self.not_alnum = {}
        self.non_ascii = {}

        self.links = {}
        self.hashtags = {}
        

        for sent_index, sentence in enumerate(sentences_list):
            entities = entities_list[sent_index]
            for w_index, word in enumerate(sentence):
                entity = entities[w_index]
                word = word.strip()
                entity = entity.strip()

                self.unique_words.add(word)
                if entity not in self.unique_entities:self.unique_entities[entity] = 1
                else: self.unique_entities[entity] += 1

                if word not in self.ambiguous_words: self.ambiguous_words[word] = [entity] # same word different entity --> Data Sanity check 
                else: self.ambiguous_words[word].append(entity)
                
                if entity not in self.word_counter:
                    self.word_counter[entity] = [word]
                    self.entity_len[entity] = [len(word)]
                else:
                    self.word_counter[entity].append(word)
                    self.entity_len[entity].append(len(word))
                
                # ---------------------------------------
                if word.istitle():
                    if entity not in self.title:self.title[entity] = [word]
                    else: self.title[entity].append(word)
                
                if word.islower():
                    if entity not in self.lower:self.lower[entity] = [word]
                    else: self.lower[entity].append(word)
                
                if word.isupper():
                    if entity not in self.abbrev_upper:self.abbrev_upper[entity] = [word]
                    else: self.abbrev_upper[entity].append(word)

                if word.isalpha(): # pure text
                    if entity not in self.alpha:self.alpha[entity] = [word]
                    else: self.alpha[entity].append(word)
                
                # -------------------------------------------- Problamatic specially it it's one of the Non O" features
                if word.isnumeric(): # digit
                    if entity not in self.nums:self.nums[entity] = [(word, sent_index, w_index)]
                    else: self.nums[entity].append((word, sent_index, w_index))
                
                if not word.isalnum(): # non alphanumeric #hashtag
                    if entity not in self.not_alnum:self.not_alnum[entity] = [(word, sent_index, w_index)]
                    else: self.not_alnum[entity].append((word, sent_index, w_index))
                
                if not word.isascii():
                    if entity not in self.non_ascii:self.non_ascii[entity] = [(word, sent_index, w_index)]
                    else: self.non_ascii[entity].append((word, sent_index, w_index))

                # -----------
                if word.startswith("http"):
                    if entity not in self.links:self.links[entity] = [(word, sent_index, w_index)]
                    else: self.links[entity].append((word, sent_index, w_index))
                
                if word.startswith("#") and (len(word) > 1) and word[1].isalnum():
                    if entity not in self.hashtags:self.hashtags[entity] = [(word, sent_index, w_index)]
                    else: self.hashtags[entity].append((word, sent_index, w_index))
    

    def create_entity_df(self):
        """
        """
        _dic = {"title":self.title,
        "lower":self.lower,
        "upper":self.abbrev_upper,
        "alpha":self.alpha}


        data = {}
        for feature_name, feature_dict in _dic.items():
            temp_dict = {}
            for ent_name, ent_arr in feature_dict.items():
                if ent_name == "O": continue
                val = len(ent_arr)
                data[f"{feature_name}_{ent_name}"] = val

                temp_dict[ent_name] = val
            
            for ent_name, ent_arr in build_parent_level_dict(temp_dict, "add").items():
                if ent_name == "O": continue
                data[f"{feature_name}_{ent_name}"] = val

        return data
    
    def check_anoomaly(self):
        """
        """
        {"nums": self.nums,
        "non_alpha":self.not_alnum,
        "non_ascii":self.non_ascii,
        "links":self.links,
        "hashtags":self.hashtags}

