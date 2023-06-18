import numpy as np
from nltk.tokenize import word_tokenize
import pickle
from collections import Counter

def preprocess(sentences, min_freq=1):
    vocabs = set()
    tokenized_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        tokenized_senetence = word_tokenize(sentence)
        tokenized_senetence.insert(0, "<s>")
        tokenized_senetence.append("</s>")
        # Collect the vocabs
        for token in tokenized_senetence:
            vocabs.add(token)
        tokenized_sentences.append(tokenized_senetence)
    # Filter out low-frequency tokens from the vocabulary
    counter = Counter(token for sentence in tokenized_sentences for token in sentence)
    for token, freq in counter.items():
        if freq < min_freq:
            vocabs.remove(token)
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