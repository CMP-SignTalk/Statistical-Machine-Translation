import dill
import json

def load_model(filepath):
    with open(filepath, 'rb') as f:
        model = dill.load(f)
    return model

def load_data(path_to_data, direction = 'forward' ,num_of_sentences = None):
    with open(path_to_data) as file:
        data = json.load(file)
    
    if num_of_sentences is not None: data = data[:num_of_sentences]
    
    f_corpus = []
    e_corpus = []
    if direction == 'forward':
        for data_item in data:
            f_corpus.append(data_item["en"])
            e_corpus.append(data_item["asl"])
    elif direction == 'backward':
        for data_item in data:
            f_corpus.append(data_item["asl"])
            e_corpus.append(data_item["en"])
    else:
        print("Invalid direction")
    return f_corpus, e_corpus 
    