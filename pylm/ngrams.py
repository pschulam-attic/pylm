from collections import Counter

class NGramStats(object):
    '''Maintains counts of n-grams.'''
    def __init__(self, order):
        self.order = order
        self.counts = Counter()

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
        return 'NGramStats(order={self.order}, #ngrams={n})'.format(self=self, n=len(self.counts))

def from_sequence(seq, vocab, order):
    '''from_sequence(seq, order) -> list of n-grams for this sequence

    Pads the beginning of the sequence with order-1 start of sentence
    markers, and 1 end of sentence marker. These markers are found in
    the vocab passed as an argument.

    '''
    sos, eos, _ = vocab.reserved_tokens()
    sos_idx, eos_idx = vocab[sos], vocab[eos]
    seq = list(seq)
    seq.append(eos_idx)
    seq = ([sos_idx]*(order-1)) + seq

    ngrams = []
    for i in xrange(len(seq)-order):
        ngrams.append(tuple(seq[i:i+order]))
    return ngrams

def compute_stats(corpus, vocab, order):
    '''compute_stats(corpus, vocab, order) -> NGramStats with counts of ngrams of order from corpus'''
    stats = NGramStats(order)
    for doc in corpus:
        for ngram in from_sequence(doc, vocab, order):
            stats.increment(ngram)
    return stats
