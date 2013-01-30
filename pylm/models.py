import math
import common
import ngrams
import smoothing

class NGramModel(object):
    '''Computes log (base 2) probabilities of token sequences.'''
    def __init__(self, order):
        self.order = order

    def train(self, corpus, vocab):
        '''model.train(corpus, vocab) -> estimate conditional probabilities over the corpus'''
        self.vocab = vocab
        self._log_probs = smoothing.additive_smoothing(corpus, vocab, self.order)
        
    def logl(self, sequence):
        '''model.logl(sequence) -> log (base 2) likelihood of the sequence'''
        ll = 0.0
        for ngram in ngrams.from_sequence(sequence, self.vocab, self.order):
            ll += self._log_probs[ngram]
        return ll
