import numpy as np
from collections import Counter

class NGramCounts(object):
    '''Maintains counts of n-grams.'''
    def __init__(self, vocab, order, storage_type=np.uint16):
        self.order = order
        self.V = len(vocab)
        dimension = tuple([V] * order)
        self.counts = np.zeros(dimension, dtype=storage_type)

    def count(self, ngram_iterable):
        '''ngramcounts.count(ngram_iterable) -> count all ngrams in this stream'''
        for ngram in ngram_iterable:
            self.increment(ngram)

    def increment(self, ngram):
        '''stats.increment(ngram) -> increment occurrences of ngram'''
        assert isinstance(ngram, tuple)
        assert len(ngram) == self.order
        self.counts[ngram] += 1

    def decrement(self, ngram):
        '''stats.decrement(ngram) -> decrement occurrences of ngram'''
        assert isinstance(ngram, tuple)
        assert len(ngram) == self.order
        assert self.counts[ngram] > 0
        self.counts[ngram] -= 1

    def __getitem__(self, ngram):
        assert isinstance(ngram, tuple)
        assert len(ngram) == self.order
        return self.counts[ngram]

    def __repr__(self):
        return 'NGramStats(vocab_size={self.V}, order={self.order}, total_count={total})'.format(self=self, n=len(self.counts), total=np.sum(self.counts))
