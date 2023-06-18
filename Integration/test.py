import tm as tm
import lm as lm
import decoder as decoder
import utils as utils
f_corpus = utils.load_data("data/small/aslg.en")
e_corpus = utils.load_data("data/small/aslg.processed.gloss.asl")
f_vocabs, f_corpus = utils.preprocess(f_corpus)
e_vocabs, e_corpus = utils.preprocess(e_corpus)
ibm1 = tm.IBMModel1(f_corpus, e_corpus, num_iters=5)
ngram_lm = lm.LM(e_vocabs, e_corpus)
ngram_lm.train()
d = decoder.Decoder(ibm1, ngram_lm)
tm_translation = d.tm_decode("my name is yousef")
print('tm_translation: ', tm_translation)
greedy_translation = d.greedy_decode("my name is yousef")
print('greedy_translation: ', greedy_translation)
beam_search_translation = d.beam_search_decode("my name is yousef")
print('beam_search_translation: ', beam_search_translation)