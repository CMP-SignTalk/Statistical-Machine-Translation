from collections import defaultdict
from nltk.tokenize import word_tokenize

class IBMModel1:
    def __init__(self, f_corpus, e_corpus):
        self.f_corpus = f_corpus
        self.e_corpus = e_corpus
        self.f_vocab = set()
        self.e_vocab = set()
   
    def _preprocess(self, sentences):
        vocabs = set()
        tokenized_sentences = []
        for sentence in sentences:
            sentence =  sentence.lower()
            tokenized_senetence = word_tokenize(sentence)
            for token in tokenized_senetence:
                vocabs.add(token)
            tokenized_sentences.append(tokenized_senetence) 
        return vocabs, tokenized_sentences

    def preprocess(self):
        self.f_vocab, self.f_corpus = self._preprocess(self.f_corpus)
        self.e_vocab, self.e_corpus = self._preprocess(self.e_corpus)

    # Fill the tranlsation table with the translation probabilities.
    def train(self, num_iters=5):
        self.translation_table = defaultdict(lambda: defaultdict(lambda: 1 / len(self.e_vocab)))
        # Run EM algorithm for num_iters iterations
        for _ in range(num_iters):
            count = defaultdict(lambda: defaultdict(float))
            total = defaultdict(float)
            s_total = defaultdict(float)
            # Expectation step
            for (i, (f, e)) in enumerate(zip(self.f_corpus, self.e_corpus)):
                for e_j in e:
                    for f_i in f:
                        s_total[e_j] += self.translation_table[f_i][e_j]
                for e_j in e:
                    for f_i in f:
                        c = self.translation_table[f_i][e_j] / s_total[e_j]
                        count[f_i][e_j] += c
                        total[f_i] += c
            # Maximization step
            for f_i in self.f_vocab:
                for e_j in self.e_vocab:
                    self.translation_table[f_i][e_j] = count[f_i][e_j] / total[f_i]
    
    def print_ds(self):
        print("f_corpus: ", self.f_corpus)
        print(' ')
        print("e_corpus: ", self.e_corpus)
        print(' ')
        print("f_vocab: ", self.f_vocab)
        print(' ')
        print("e_vocab: ", self.e_vocab)
        print(' ')
        print("translation_table: ", self.translation_table)
        print(' ')