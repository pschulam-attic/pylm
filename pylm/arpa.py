import math
import os
import subprocess as sp
import kenlm

from tempfile import mktemp

class ArpaLM(object):
    def __init__(self, path):
        self.lm = kenlm.LanguageModel(path)
        self.log10_likelihood = 0.0

    def prob(self, context, word):
        sentence = ' '.join(context) + ' ' + word
        return 10**self.lm.full_scores(sentence)[-1][0]

    def log_likelihood(self, sentence):
        return self.lm.score(' '.join(sentence)) / math.log(2, 10)

    def __repr__(self):
        return 'ArpaLM(order={self.lm.order})'.format(self=self)

class CharLM(ArpaLM):
    def prob(self, word):
        return 10**self.lm.score(' '.join(word))

    def increment(self, word):
        pass

    def decrement(self, word):
        pass

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

    @property
    def ngram_file(self):
        return self._ngram

    @property
    def vocab_file(self):
        return self._vocab

    def write_ngram(self, out_stream):
        with open(self._ngram) as ngram:
            out_stream.write(ngram.read())

    def write_vocab(self, out_stream):
        with open(self._vocab) as vocab:
            out_stream.write(vocab.read())
