import utils as utils
import lm as lm

path_to_data = "data.txt"
training_data = utils.load_data(path_to_data)
vocabs, preprocessed_training_data = utils.preprocess(training_data)

# ilm = lm.LM(vocabs, preprocessed_training_data)
# ilm.train()

# sentence_probability = ilm.calc_sentence_probability('She is a teacher.')
# print('sentence_probability',sentence_probability)
# log_sentence_probability = ilm.calc_log_sentence_probability('She is a teacher.')
# print('log_sentence_probability',log_sentence_probability)


unigram_lm = lm.Unigram(vocabs, preprocessed_training_data)
unigram_lm.train()
# probability = unigram_lm.calc_probability('is')
# print('probability',probability)
# sentence_probability = unigram_lm.calc_sentence_probability('she is a teacher.')
# print('sentence_probability',sentence_probability)
# log_probability = unigram_lm.calc_log_probability('is')
# print('log_probability',log_probability)
# log_sentence_probability = unigram_lm.calc_log_sentence_probability('she is a teacher.')
# print('log_sentence_probability',log_sentence_probability)

probability = unigram_lm.calc_probability('cmp')
print('probability',probability)
sentence_probability = unigram_lm.calc_sentence_probability('she is a cmp teacher.')
print('sentence_probability',sentence_probability)