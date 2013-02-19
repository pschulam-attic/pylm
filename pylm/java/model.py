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
