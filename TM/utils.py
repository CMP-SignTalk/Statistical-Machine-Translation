import numpy as np
from nltk.tokenize import word_tokenize
import pickle

# Takes a list of sentences, preprocess them and collects the vocabularies.
def preprocess(sentences):
    vocabs = set()
    tokenized_sentences = []
    for sentence in sentences:
        sentence =  sentence.lower()
        tokenized_senetence = word_tokenize(sentence)
        tokenized_senetence.insert(0, "<s>")
        tokenized_senetence.insert(len(tokenized_senetence), "</s>")
        # Collect the vocabs
        for token in tokenized_senetence:
           vocabs.add(token)
        tokenized_sentences.append(tokenized_senetence) 
    return list(vocabs), tokenized_sentences

def load_data(path_to_data):
    with open(path_to_data,'r') as f:
        data = [line.strip() for line in f.readlines()]
    return data

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model