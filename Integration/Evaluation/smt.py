import os  
import sys

# Add the Integration directory to the Python system path, enabling the import of modules or packages located in that directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils as utils
import Decoder.ibm1 as ibm1_decoder
import Decoder.phrase_based as phrase_based_decoder

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

class SMT:
    def __init__(self, laguage_model , ibm1_translation_model, phrass_based_translation_model):
        self.laguage_model = laguage_model
        self.ibm1_translation_model = ibm1_translation_model
        self.phrass_based_translation_model = phrass_based_translation_model

        # Get the absolute path of the current directory
        folder_dir = os.path.dirname(os.path.abspath(__file__))

        # Load the language model
        self.lm = utils.load_model(os.path.join(folder_dir, 'models', self.laguage_model + '.pkl'))

        # Load the translation models both the IBM1 and the Phrase-Based
        self.ibm1 = utils.load_model(os.path.join(folder_dir, 'models', self.ibm1_translation_model + '.pkl'))
        self.phrase_based = utils.load_model(os.path.join(folder_dir, 'models', self.phrass_based_translation_model + '.pkl'))

        # Instantiate the decoders
        self.ibm1_translator = ibm1_decoder.Decoder(self.ibm1, self.lm)
        self.phrase_based_translator = phrase_based_decoder.Decoder(self.phrase_based, self.lm)

    def translate(self, f):
        e = self.phrase_based_translator.translate(f)
        if  not e : # If the phrase based decoder fails to translate, use the IBM1 decoder
            e = self.ibm1_translator.translate(f)
        return e
    
    
    # Evaluate the translation using BLEU score
    def evaluate(self, f_corpus, e_corpus):
        # Translate the f_corpus to e_corpus_hat - Now we have the references (e_corpus) and the hypotheses (e_corpus_hat)
        e_corpus_hat = [self.translate(f) for f in f_corpus]
        print('e_corpus_hat', e_corpus_hat)

        # Split each sentence in e_corpus_hat into a list of words and store them in hypotheses
        hypotheses = [e_hat.split() for e_hat in e_corpus_hat]
        print('hypotheses', hypotheses)

        # Split each sentence in e_corpus into a list of list of words and store them in references
        references = [[e.split()] for e in e_corpus]
        print('references', references)

        # Calculate the corpus BLEU score
        sf = SmoothingFunction()
        return corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),  smoothing_function = sf.method1) * 100


