from utils import *
# Import the decoders
import ibm1 as ibm1
import phrase_based as phrase_based

# Load the language models 
bigram_lm_forward = load_model('models/lm/bigram_lm_forward.pkl')
bigram_lm_backward = load_model('models/lm/bigram_lm_backward.pkl')

# Load the translation models
ibm1_forward = load_model('models/tm/ibm1_forward.pkl')
ibm1_backward = load_model('models/tm/ibm1_backward.pkl')
phrase_based_forward = load_model('models/tm/phrase_based_forward.pkl')
phrase_based_backward = load_model('models/tm/phrase_based_backward.pkl')

# Instantiate the decoders
ibm1_forward_translator = ibm1.Decoder(ibm1_forward, bigram_lm_forward)
ibm1_backward_translator = ibm1.Decoder(ibm1_backward, bigram_lm_backward)
phrase_based_forward_translator = phrase_based.Decoder(phrase_based_forward, bigram_lm_forward)
phrase_based_backward_translator = phrase_based.Decoder(phrase_based_backward, bigram_lm_backward)

def forward_translate(f):
    e = phrase_based_forward_translator.translate(f)
    if  not e : # If the phrase based decoder fails to translate, use the IBM1 decoder
        e = ibm1_forward_translator.translate(f)
    return e

def backward_translate(f):
    e = phrase_based_backward_translator.translate(f)
    if  not e : # If the phrase based decoder fails to translate, use the IBM1 decoder
        e = ibm1_backward_translator.translate(f)
    return e

aslg = forward_translate('the girl is in france')
print(aslg)

en = backward_translate(aslg)
print(en)

aslg = forward_translate('the girl in france')
print(aslg)

en = backward_translate(aslg)
print(en)