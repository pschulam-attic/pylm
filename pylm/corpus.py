import gzip
import numpy as np

class Vocab(object):
    '''Maps strings to integers and vice versa.'''
    def __init__(self, sos='<s>', eos='</s>', unk='<unk>'):
        self._word_to_int = {}
        self._int_to_word = []
        self._frozen = False
        self._reserved = (sos, eos, unk)
        for w in self._reserved:
            self[w]

    def freeze(self):
        '''vocab.freeze() -> no more words can be added'''
        self._frozen = True

    def reserved_tokens(self):
        '''vocab.reserved_tokens() -> (sos, eos, unk)
        
        where:
            - sos is the start of sentence marker for this vocab
            - eos is the eos of sentence marker for this vocab
            - unk is the unk marker for this vocab

        '''
        return self._reserved

    def sos(self):
        return self._reserved[0]

    def eos(self):
        return self._reserved[1]
    
    def unk(self):
        return self._reserved[2]

    def __getitem__(self, key):
        '''vocab[string] -> integer | vocab[index] -> string'''
        wti = self._word_to_int
        itw = self._int_to_word
        unk = self.unk()
        if isinstance(key, str):
            if self._frozen:
                return wti[unk]
            elif key not in wti:
                wti[key] = len(self)
                itw.append(key)
            return wti[key]
        elif isinstance(key, int):
            if key >= len(self) or key < 0:
                raise IndexError('invalid word ID: {0}'.format(key))
            else:
                return itw[key]
        else:
            raise TypeError('vocab key must be str or int')

    def __len__(self):
        '''len(vocab) -> number of words currently indexed'''
        return len(self._int_to_word)

    def __repr__(self):
        return 'Vocab(#words={0}, frozen={self._frozen})'.format(len(self), self=self)

class Corpus(object):
    '''Contains data read from a corpus file.'''
    def __init__(self, readable_object, vocab=None, storage_type=np.uint16):
        '''reads lines from readable_object and splits on whitespace
        to get words. If no vocab is given, a new vocab is created and
        stored with this corpus.

        '''
        self._vocab = vocab if vocab else Vocab()
        self._data = []
        for line in readable_object:
            words = line.strip().split()
            a = np.array([self._vocab[w] for w in words], dtype=storage_type)
            self._data.append(a)
        if not vocab:
            self._vocab.freeze()
        self._ntokens = sum(len(a) for a in self._data)

    def ngrams(self, order):
        '''corpus.ngrams(order) -> iterable over all ngrams of order from this corpus

        This method generates the ngrams lazily, so if you want them
        all at once, you should explicitly convert the iterable to a
        list.

        '''
        for doc in self._data:
            words = list(doc)
            v = self._vocab
            words.append(v[v.eos()])
            words = ([v[v.sos()]] * (order-1)) + words
            for i in xrange(len(words)-order):
                yield tuple(words[i:i+order])

    def num_types(self):
        return len(self._vocab)

    def num_tokens(self):
        return self._ntokens
    
    def num_docs(self):
        return len(self._data)

    def spawn_corpus(self, readable_object):
        '''corpus.spawn_corpus(readable_object) -> a new Corpus object with the same vocab but different data'''
        return Corpus(readable_object, vocab=self._vocab)

    def __repr__(self):
        return 'Corpus(#documents={0}, #tokens={1}, #types={2})'.format(len(self._data), self._ntokens, len(self._vocab))

def from_txt(filename):
    '''corpus_from_text(filename) -> new corpus object'''
    with open(filename) as f:
        c = Corpus(f)
    return c

def from_gzip(filename):
    '''corpus_from_gzip(filename) -> new corpus object'''
    with gzip.open(filename) as f:
        c = Corpus(f)
    return c
