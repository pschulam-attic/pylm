import itertools
import ngrams
import common

def additive_smoothing(corpus, vocab, order, delta=0.5):
    joint_stats = ngrams.compute_stats(corpus, vocab, order)
    evidence_stats = ngrams.compute_stats(corpus, vocab, order-1)
    V = len(vocab)
    log_probs = {}
    l2 = common.log2
    id_iterators = itertools.tee(xrange(V), order)
    possible_ngrams = itertools.product(*id_iterators)
    for ngram in possible_ngrams:
        log_probs[ngram] = (l2(delta + joint_stats[ngram]) -
                            l2(V * delta + evidence_stats[ngram[:-1]]))
    return log_probs
