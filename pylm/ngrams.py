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
