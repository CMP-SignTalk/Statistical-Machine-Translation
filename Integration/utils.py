from nltk.tokenize import word_tokenize
import dill
import json

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

def load_data(path_to_data, num_of_sentences = None):
    with open(path_to_data) as file:
        data = json.load(file)
    
    if num_of_sentences is not None: data = data[:num_of_sentences]

    en=[]
    asl=[]
    for data_item in data:
        en.append(data_item["en"])
        asl.append(data_item["asl"])

    return en, asl
    
def save_model(model, filename):
    with open(filename, 'wb') as f:
        dill.dump(model, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        model = dill.load(f)
    return model