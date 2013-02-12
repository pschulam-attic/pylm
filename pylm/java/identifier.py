import math
from collections import defaultdict, deque
from ..models import DirichletCategorical, Categorical
from ..interpolate import get_model_weights
from utils import read_token_info

def read_identifiers(filename):
    tokens = (info for seq in read_token_info(filename) for info in seq)
    for t, p, id_type, is_dec in tokens:
        if p == 'Identifier':
            yield t, id_type, is_dec

class TypedIdentifierModel(object):
    def __init__(self):
        self._trained = False

    def train(self, id_info_tuples):
        id_info_tuples = list(id_info_tuples)
        self._types = set(t for _, t, _ in id_info_tuples)
        type_vocabs = {}
        for this_type in self._types:
            vocab = set(i for i, t, _ in id_info_tuples if t == this_type)
            type_vocabs[this_type] = vocab
        self._type_models = {}
        for t, v in type_vocabs.items():
            self._type_models[t] = DirichletCategorical(v, 0.5)

        for i, t, _ in id_info_tuples:
            self._type_models[t].increment(i)

    def prob(self, id_info):
        i, t, _ = id_info
        return self._type_models[t].prob(i)

    def probs(self, id_info_tuples):
        return [self.prob(info) for info in id_info_tuples]

class DeclarationAwareModel(object):
    def __init__(self, typed_model):
        self._typed_model = typed_model
        self._type_dec_models = defaultdict(lambda: Categorical())

    def probs(self, id_info_tuples):
        log_prob = 0.0
        to_interpolate = []
        for i, t, is_dec in id_info_tuples:
            is_dec = True if is_dec == 'true' else False
            if is_dec:
                self._type_dec_models[t].increment(i)
                log_prob += math.log(self._typed_model.prob((i, t, is_dec)), 10)
            else:
                p1 = self._typed_model.prob((i, t, is_dec))
                p2 = self._type_dec_models[t].prob(i)
                to_interpolate.append( (p1, p2) )
        
        m1, m2 = zip(*to_interpolate)
        w1, w2 = get_model_weights(m1, m2)
        print 'weights: {} {}'.format(w1, w2)
        for l1, l2 in zip(m1, m2):
            log_prob += math.log(w1 * l1 + w2 * l2, 10)
        return log_prob

    def __repr__(self):
        return repr(self._type_dec_models.items())

class CacheModel(object):
    def __init__(self, size=50):
        self._size = size
    
    def probs(self, id_info_tuples):
        probs = []
        cache = deque(maxlen=self._size)
        for i, t, is_dec in id_info_tuples:
            if len(cache) == 0:
                cache.append(i)
                probs.append(0.0)
            else:
                p = float(cache.count(i)) / len(cache)
                probs.append(p)
                cache.append(i)
        return probs

class TypedCacheModel(object):
    def __init__(self, size=20):
        self._size = size
        
    def probs(self, id_info_tuples):
        probs = []
        caches = defaultdict(lambda: deque(maxlen=self._size))
        for i, t, is_dec in id_info_tuples:
            if len(caches[t]) == 0:
                caches[t].append(i)
                probs.append(0.0)
            else:
                p = float(caches[t].count(i)) / len(caches[t])
                probs.append(p)
                caches[t].append(i)
        return probs
