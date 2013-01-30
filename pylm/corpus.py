import gzip
import itertools

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

    def __getitem__(self, key):
        '''vocab[string] -> integer | vocab[index] -> string'''
        wti = self._word_to_int
        itw = self._int_to_word
        if isinstance(key, str):
            if key not in wti:
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

def from_txt(filename):
    '''corpus_from_text(filename) -> (corpus, vocab)

    Corpus is a list of lists of word IDs. Word IDs can be looked up
    in vocab.

    '''
    v = Vocab()
    c = []
    for line in open(filename):
        words = line.strip().split()
        c.append([v[w] for w in words])
    return c, v

def from_gzip(filename):
    '''corpus_from_gzip(filename) -> (corpus, vocab)

    Corpus is a list of lists of word IDs. Word IDs can be looked up
    in vocab.

    '''
    v = Vocab()
    c = []
    for line in gzip.open(filename):
        words = line.strip().split()
        c.append([v[w] for w in words])
    return c, v
