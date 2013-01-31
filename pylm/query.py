from external.kenlm import LanguageModel

class ArpaQueryer(object):
    '''Answers probability queries using an arpa n-gram model.'''
    def __init__(self, arpa_filepath, vocab_file):
        self._filename = arpa_filepath
        self._lm = LanguageModel(arpa_filepath)
        self._vocab = set(l.strip() for l in open(vocab_file))

    def prob_of_sentence(self, sentence):
        '''query.prob_of_sentence(sentence) -> log base 10 probability of sentence'''
        return self._lm.score(sentence)

    def full_prob_of_sentence(self, sentence):
        '''query.full_prob_of_sentence(sentence) -> iterable over 2-tuples (prob, n-gram length)'''
        return self._lm.full_scores(sentence)

    def cond_distribution(self, prefix):
        '''query.prob_of_next(prefix) -> distribution over vocab items given this prefix'''
        dist = {}
        for w in self._vocab:
            probs = list(self.full_prob_of_sentence(prefix + ' ' + w))
            dist[w] = probs[-2][0]
        return dist
            
