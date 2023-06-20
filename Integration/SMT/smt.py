# from nltk.tokenize import word_tokenize
# import tm as tm
# import lm as lm
# import decoder as decoder

# class SMT:
#     def __init__(self, f_corpus, e_corpus, tm=None, num_iters=5, lm=None, decoder=None):
#       self.f_vocabs, self.f_corpus = self.preprocess(f_corpus)
#       self.e_vocabs, self.e_corpus = self.preprocess(e_corpus)
#       # Initialize the selected translation model.
#       self.ibm1 = tm.IBMModel1(f_corpus, e_corpus, num_iters=5)
#       # Initialize the selected language model.

#     def preprocess(sentences):
#       vocabs = set()
#       tokenized_sentences = []
#       for sentence in sentences:
#           sentence =  sentence.lower()
#           tokenized_senetence = word_tokenize(sentence)
#           tokenized_senetence.insert(0, "<s>")
#           tokenized_senetence.insert(len(tokenized_senetence), "</s>")
#           # Collect the vocabs
#           for token in tokenized_senetence:
#             vocabs.add(token)
#           tokenized_sentences.append(tokenized_senetence) 
#       return list(vocabs), tokenized_sentences

#     def train(self):
#       # Train the translation model.
#       # Train the language model.
#       pass

#     def translate(self, f):
#       # Implement the decoding algorithm.
#       pass