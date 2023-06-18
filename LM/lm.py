import math
import numpy as np
from nltk import bigrams, trigrams, FreqDist
from nltk.tokenize import word_tokenize
from collections import Counter

class Unigram:
    def __init__(self, corpus):
        self.corpus = corpus

    def preprocess(self, min_freq=1):
        self.vocabs = set()
        self.sentences = []
        for sentence in self.corpus:
            sentence = sentence.lower()
            sentence = word_tokenize(sentence)
            sentence.insert(0, "<s>")
            sentence.append("</s>")
            # Collect the vocabs
            for token in sentence:
                self.vocabs.add(token)
            self.sentences.append(sentence)
        # Filter out low-frequency tokens from the vocabulary
        counter = Counter(token for sentence in self.sentences for token in sentence)
        for token, freq in counter.items():
            if freq < min_freq:
                self.vocabs.remove(token)
        self.vocabs = list(self.vocabs)

        # Add the <unk> token to handle unknown words -> words that are out of the vocabulary of the language model.
        self.vocabs.append("<unk>")
        self.vocabs_length = len(self.vocabs)
        self.id2word = {i: word for i, word in enumerate(list(self.vocabs))}
        self.word2id = {word: i for i, word in self.id2word.items()}
        self.counts = np.zeros(self.vocabs_length, dtype=int)

    # Fill the 1D array with the appropriate counts.
    def train(self):
        for sentence in self.sentences:
            for word in sentence:
                if word in self.vocabs:
                    self.counts[self.word2id[word]] += 1
                else: # unknown word
                    self.counts[self.word2id["<unk>"]] += 1

    # Calculates the add-k smoothed probabilities - To handle the unseen words.
    def calc_probability(self, word, k=1):
        if word in self.vocabs: # The word is in the vocabulary of the language model. 
            return (self.counts[self.word2id[word]] + k) / (np.sum(self.counts) + k*self.vocabs_length)
        else:
            return k / (np.sum(self.counts) + k*self.vocabs_length + self.counts[self.word2id["<unk>"]])
        
    def calc_sentence_probability(self, sentence):
        sentence =  sentence.lower()
        sentence = word_tokenize(sentence)
        sentence.insert(0, "<s>")
        sentence.insert(len(sentence), "</s>")
        probability = 1
        for i in range(0, len(sentence)):
            probability *= self.calc_probability(sentence[i])            
        return probability

    def calc_log_probability(self, word):
        return np.log(self.calc_probability(word))

    def calc_log_sentence_probability(self, sentence):
        sentence =  sentence.lower()
        sentence = word_tokenize(sentence)
        sentence.insert(0, "<s>")
        sentence.insert(len(sentence), "</s>")
        probability = 0
        for i in range(0, len(sentence)):
            probability += self.calc_log_probability(sentence[i])       
        return probability
    
    def perplexity(self, test_corpus):
        test_sentences = []
        for sentence in test_corpus:
            sentence = sentence.lower()
            sentence = word_tokenize(sentence)
            sentence.insert(0, "<s>")
            sentence.append("</s>")
            test_sentences.append(sentence)
        
        N = 0 # total number of tokens
        log_likelihood = 0
        for sentence in test_sentences:
            for word in sentence:
                N += 1
                log_likelihood += self.calc_log_probability(word)
                
        perplexity = math.exp(-log_likelihood / N)
        return perplexity
    
###########################################################################################################

class Bigram:
    def __init__(self, corpus):
        self.corpus = corpus

    def preprocess(self, min_freq=1):
        self.vocabs = set()
        self.sentences = []
        for sentence in self.corpus:
            sentence = sentence.lower()
            sentence = word_tokenize(sentence)
            sentence.insert(0, "<s>")
            sentence.append("</s>")
            # Collect the vocabs
            for token in sentence:
                self.vocabs.add(token)
            self.sentences.append(sentence)
        # Filter out low-frequency tokens from the vocabulary
        counter = Counter(token for sentence in self.sentences for token in sentence)
        for token, freq in counter.items():
            if freq < min_freq:
                self.vocabs.remove(token)
        self.vocabs = list(self.vocabs)

        # Add the <unk> token to handle unknown words -> words that are out of the vocabulary of the language model.
        self.vocabs.append("<unk>")
        self.vocabs_length = len(self.vocabs)
        self.id2word = {i: word for i, word in enumerate(list(self.vocabs))}
        self.word2id = {word: i for i, word in self.id2word.items()}
        self.counts = np.zeros((self.vocabs_length, self.vocabs_length), dtype=int)

    # Fill the 2D array with the appropriate counts
    def train(self):
        for sentence in self.sentences:
            sentence_bigrams = bigrams(sentence)
            sentence_bigrams_fd = FreqDist(sentence_bigrams)
            for bigram , frequency in sentence_bigrams_fd.items():
                if bigram[0] in self.vocabs and bigram[1] in self.vocabs:
                    self.counts[self.word2id[bigram[0]], self.word2id[bigram[1]]] += frequency                
                else:
                    self.counts[self.word2id["<unk>"], self.word2id["<unk>"]] += frequency

    def calc_probability(self, word1, word2, k=1):
        if (word1 in self.vocabs) and (word2 in self.vocabs):
            return (self.counts[self.word2id[word1], self.word2id[word2]] + k)/(np.sum(self.counts[self.word2id[word1], :]) + k*self.vocabs_length)
        else:
            return k / (np.sum(self.counts[self.word2id["<unk>"], :]) + k*self.vocabs_length + self.counts[self.word2id["<unk>"], self.word2id["<unk>"]])

    def calc_sentence_probability(self, sentence):
        sentence =  sentence.lower()
        sentence = word_tokenize(sentence)
        sentence.insert(0, "<s>")
        sentence.insert(len(sentence), "</s>")
        probability = 1
        for i in range(1, len(sentence)):
            probability *= self.calc_probability(sentence[i-1], sentence[i])           
        return probability
    
    def calc_log_probability(self, word1, word2):
        return np.log(self.calc_probability(word1, word2))
    
    def calc_log_sentence_probability(self, sentence):
        sentence =  sentence.lower()
        sentence = word_tokenize(sentence)
        sentence.insert(0, "<s>")
        sentence.insert(len(sentence), "</s>")
        probability = 0
        for i in range(1, len(sentence)):
            probability += self.calc_log_probability(sentence[i-1], sentence[i])       
        return probability

    def perplexity(self, test_corpus):
        test_sentences = []
        for sentence in test_corpus:
            sentence = sentence.lower()
            sentence = word_tokenize(sentence)
            sentence.insert(0, "<s>")
            sentence.append("</s>")
            test_sentences.append(sentence)
        
        N = 0 # total number of bigrams
        log_likelihood = 0
        for sentence in test_sentences:
            for i in range(1, len(sentence)):
                N += 1
                log_likelihood += self.calc_log_probability(sentence[i-1], sentence[i])
        
        perplexity = math.exp(-log_likelihood / N)
        return perplexity
    
###########################################################################################################

class Trigram:
    def __init__(self, corpus):
        self.corpus = corpus

    def preprocess(self, min_freq=1):
        self.vocabs = set()
        self.sentences = []
        for sentence in self.corpus:
            sentence = sentence.lower()
            sentence = word_tokenize(sentence)
            sentence.insert(0, "<s>")
            sentence.append("</s>")
            # Collect the vocabs
            for token in sentence:
                self.vocabs.add(token)
            self.sentences.append(sentence)
        # Filter out low-frequency tokens from the vocabulary
        counter = Counter(token for sentence in self.sentences for token in sentence)
        for token, freq in counter.items():
            if freq < min_freq:
                self.vocabs.remove(token)
        self.vocabs = list(self.vocabs)

        # Add the <unk> token to handle unknown words -> words that are out of the vocabulary of the language model.
        self.vocabs.append("<unk>")
        self.vocabs_length = len(self.vocabs)
        self.id2word = {i: word for i, word in enumerate(list(self.vocabs))}
        self.word2id = {word: i for i, word in self.id2word.items()}
        vocab_pairs = [(v1, v2) for v1 in list(self.vocabs) for v2 in list(self.vocabs)]
        self.id2pair = {i: pair for i, pair in enumerate(vocab_pairs)}
        self.pair2id = {pair: i for i, pair in self.id2pair.items()}
        self.counts = np.zeros((self.vocabs_length * self.vocabs_length, self.vocabs_length), dtype=int)

    # Fill the 2D matrix with the appropriate counts.
    def train(self):
        for sentence in self.sentences:
            sentence_trigrams = trigrams(sentence)
            sentence_trigrams_fd = FreqDist(sentence_trigrams)
            for trigram, frequency in sentence_trigrams_fd.items():
                if trigram[0] in self.vocabs and trigram[1] in self.vocabs and trigram[2] in self.vocabs:
                    self.counts[self.pair2id[(trigram[0], trigram[1])], self.word2id[trigram[2]]] += frequency                
                else:
                    self.counts[self.pair2id[("<unk>", "<unk>")], self.word2id["<unk>"]] += frequency

    def calc_probability(self, word1, word2, word3, k=1):
        if (word1, word2) in self.pair2id and word3 in self.vocabs:
            return (self.counts[self.pair2id[(word1, word2)], self.word2id[word3]] + k) / (np.sum(self.counts[self.pair2id[(word1, word2)], :]) + k*self.vocabs_length)
        else:
            return k / (np.sum(self.counts[self.pair2id[("<unk>", "<unk>")], :]) + k*self.vocabs_length + self.counts[self.pair2id[("<unk>", "<unk>")], self.word2id["<unk>"]])


    def calc_sentence_probability(self, sentence):
        sentence =  sentence.lower()
        sentence = word_tokenize(sentence)
        sentence.insert(0, "<s>")
        sentence.insert(len(sentence), "</s>")
        probability = 1
        for i in range(2, len(sentence)):
            probability *= self.calc_probability(sentence[i-2], sentence[i-1], sentence[i])       
        return probability
    
    def calc_log_probability(self, word1, word2, word3):
        return np.log(self.calc_probability(word1, word2, word3))    

    def calc_log_sentence_probability(self, sentence):
        sentence =  sentence.lower()
        sentence = word_tokenize(sentence)
        sentence.insert(0, "<s>")
        sentence.insert(len(sentence), "</s>")
        probability = 0
        for i in range(2, len(sentence)):
            probability += self.calc_log_probability(sentence[i-2], sentence[i-1], sentence[i])       
        return probability

    def perplexity(self, test_corpus):
        test_sentences = []
        for sentence in test_corpus:
            sentence = sentence.lower()
            sentence = word_tokenize(sentence)
            sentence.insert(0, "<s>")
            sentence.append("</s>")
            test_sentences.append(sentence)
        
        N = 0
        log_likelihood = 0
        for sentence in test_sentences:
            for i in range(2, len(sentence)):
                N += 1
                log_likelihood += self.calc_log_probability(sentence[i-2], sentence[i-1], sentence[i])

        perplexity = math.exp(-log_likelihood / N)
        return perplexity
        
###########################################################################################################

class LM:
    def __init__(self, sentences):
        self.unigram_lm = Unigram(sentences)
        self.bigram_lm = Bigram(sentences)
        self.trigram_lm = Trigram(sentences)
        
    def preprocess(self, min_freq=1):
        self.unigram_lm.preprocess(min_freq)
        self.bigram_lm.preprocess(min_freq)
        self.trigram_lm.preprocess(min_freq)

    def train(self):
        self.unigram_lm.train()
        self.bigram_lm.train()
        self.trigram_lm.train()

    def calc_sentence_probability(self, sentence):
        return np.exp(self.calc_log_sentence_probability(sentence))

    def calc_log_sentence_probability(self, sentence):
        sentence =  sentence.lower()
        sentence = word_tokenize(sentence)
        sentence.insert(0, "<s>")
        sentence.insert(len(sentence), "</s>")
        probability = 0
        lambda1 = 0.2
        lambda2 = 0.3
        lambda3 = 0.5
        for i in range(2, len(sentence)):
            trigram_probability = self.trigram_lm.calc_probability(sentence[i-2], sentence[i-1], sentence[i])
            bigram_probability = self.bigram_lm.calc_probability(sentence[i-1], sentence[i])
            unigram_probability = self.unigram_lm.calc_probability(sentence[i])
            probability += np.log(lambda1 * trigram_probability + lambda2 * bigram_probability + lambda3 * unigram_probability)
        return probability
    