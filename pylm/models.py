import math
import os
import subprocess as sp
from tempfile import mktemp
from collections import Counter

import ngrams
import smoothing
import query

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

class SRILMWrapper(object):
    def __init__(self):
        self._ngram = mktemp()
        self._vocab = mktemp()
        self._trained = False
        self._query = None

    def __del__(self):
        os.remove(self._ngram)
        os.remove(self._vocab)

    def train(self, sentences, order, discount=None):
        discount = '-' + discount if discount else ''
        cmd = ('ngram-count -text - ' +
               '-order {order} ' +
               '-write-vocab {vocab} -unk ' +
               '-lm {ngram} {discount}').\
               format(order=order,
                      vocab=self._vocab,
                      ngram=self._ngram,
                      discount=discount)
        sentences = (' '.join(s) for s in sentences)
        p = sp.Popen(cmd, shell=True, stdin=sp.PIPE)
        p.communicate('\n'.join(sentences))
        self._trained = True

    def prob(self, sentence_str):
        if self._trained and not self._query:
            self._query = query.ArpaQueryer(self._ngram, self._vocab)
        return self._query.prob_of_sentence(sentence_str)

    @property
    def ngram_file(self):
        return self._ngram

    @property
    def vocab_file(self):
        return self._vocab

    def write_ngram(self, fn):
        with open(self._ngram) as fin, open(fn, 'w') as fout:
            fout.write(fin.read())

    def write_vocab(self, fn):
        with open(self._ngram) as fin, open(fn, 'w') as fout:
            fout.write(fin.read())

class Categorical(object):
    def __init__(self):
        self._counts = Counter()
        self._N = 0

    def increment(self, token, amount=1):
        self._counts[token] += amount
        self._N += amount

    def prob(self, token):
        if self._N == 0:
            return 0.0
        return float(self._counts[token]) / self._N

    def probs(self, tokens):
        return [self.prob(t) for t in tokens]

class DirichletCategorical(object):
    def __init__(self, vocab, alpha):
        self._counts = Counter()
        self._N = 0
        self._alpha = float(alpha)
        self._vocab = vocab
        self._unk_token = '<unk>'
        self._K = len(vocab) + 1

    def increment(self, token, amount=1):
        if token not in self._vocab:
            token = self._unk_token
        self._counts[token] += amount
        self._N += amount

    def prob(self, token):
        if token not in self._vocab:
            token = self._unk_token
        p = (self._alpha + self._counts[token]) / (self._K * self._alpha + self._N)
        return p

    def probs(self, tokens):
        return [self.prob(t) for t in tokens]
