import nltk

class Decoder:
    def __init__(self, translation_model, language_model = None):
        self.translation_model = translation_model
        self.language_model = language_model

    def tm_decode(self, f):
        f_tokens = nltk.word_tokenize(f.lower())

        e_tokens = []

        for token in f_tokens:
            # print('token:-------------------------------------->', token)
            translation_probs = self.translation_model.translation_table[token]
            # print('translation_probs:-------------------------------------->', translation_probs)
            # sort the translation_probs by value (probability) in descending order - for debugging purposes
            # sorted_translation_probs = sorted(translation_probs.items(), key=lambda kv: kv[1], reverse=True)
            # print('sorted_translation_probs:-------------------------------------->', sorted_translation_probs)
            if translation_probs:
                best_e_token = max(translation_probs, key=translation_probs.get) 
                # max() applies translation_probs.get to each element of the translation_probs and uses the result (probabilities) to determine the maximum value
                # the effect of key=translation_probs.get is that the max() function finds the key in the translation_probs dictionary that has the highest value, rather than finding the highest value itself
                e_tokens.append(best_e_token)
            else:
                e_tokens.append(token)

        e_tokens = [token for token in e_tokens if token is not None]
        e = ' '.join(e_tokens)
        return e
    
    # def calculate_sentence_prob(self, e_tokens, e_token):
    #     # print(' '.join(e_tokens + [e_token]), ' sentence probability is: ' , self.language_model.calc_sentence_probability(' '.join(e_tokens + [e_token])))
    #     return self.language_model.calc_sentence_probability(' '.join(e_tokens + [e_token]))
    
    def greedy_decode(self, f):
        f_tokens = nltk.word_tokenize(f.lower())

        e_tokens = []

        for token in f_tokens:
            # print('token:-------------------------------------->', token)
            translation_probs = self.translation_model.translation_table[token]
            # print('translation_probs:-------------------------------------->', translation_probs)
            # sort the translation_probs by value (probability) in descending order - for debugging purposes
            # sorted_translation_probs = sorted(translation_probs.items(), key=lambda kv: kv[1], reverse=True)
            # print('sorted_translation_probs:-------------------------------------->', sorted_translation_probs)
            if translation_probs:
                # use the language model to choose best translation token (not just the most probable)
                best_e_token = max(translation_probs, key=lambda e_token: self.language_model.calc_sentence_probability(' '.join(e_tokens + [e_token])))
                e_tokens.append(best_e_token)
            else:
                e_tokens.append(token)

        e_tokens = [token for token in e_tokens if token is not None]
        e = ' '.join(e_tokens)
        return e
    
    def beam_search_decode(self, f, beam_size=5):
        f_tokens = nltk.word_tokenize(f.lower())

        hypotheses  = [(0.0, [])]

        for token in f_tokens:
            new_hypotheses  = []
            # Generate candidate translations for each beam item
            for score, e_tokens in hypotheses:
                translation_probs = self.translation_model.translation_table.get(token)
                if translation_probs:
                    # Get the top `beam_size` target tokens for the current source token
                    e_token_candidates = sorted(translation_probs, key=translation_probs.get, reverse=True)[:beam_size]
                else:
                    # If there are no candidates, append the source token to the current beam item
                    e_token_candidates = [token]

                # Calculate the score of each candidate translation option and add it to the new beam
                for e_token_candidate in e_token_candidates:
                    e_tokens_candidate = e_tokens + [e_token_candidate]
                    score_candidate = score + self.language_model.calc_sentence_probability(' '.join(e_tokens_candidate))
                    new_hypotheses.append((score_candidate, e_tokens_candidate))

            # Keep only the top `beam_size` translations according to their scores
            hypotheses = sorted(new_hypotheses, key=lambda x: x[0], reverse=True)[:beam_size]

        # Return the top scoring translation
        return ' '.join(hypotheses[0][1])