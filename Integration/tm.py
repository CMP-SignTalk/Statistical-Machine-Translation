from collections import defaultdict

class IBMModel1:
    def __init__(self, f_corpus, e_corpus, num_iters=5):
        self.f_corpus = f_corpus
        self.e_corpus = e_corpus
        self.num_iters = num_iters
        # Collect vocabulary from the corpus.
        self.f_vocab = set()
        self.e_vocab = set()
        for (i, (f, e)) in enumerate(zip(f_corpus, e_corpus)):
            for f_i in f:
                self.f_vocab.add(f_i)
            for e_j in e:
                self.e_vocab.add(e_j)
        self.translation_table = defaultdict(lambda: defaultdict(lambda: 1 / len(self.e_vocab)))
        self.train()
        
    # Fill the tranlsation table with the translation probabilities.
    def train(self):
        # Run EM algorithm for num_iters iterations
        for _ in range(self.num_iters):
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
