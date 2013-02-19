import logging
import math

from ..pyp import PYP
from ..arpa import CharLM, SRILMWrapper

import corpus

def run_sampler(model, identifiers, n_iter):
    n_identifiers = len(identifiers)
    for it in range(n_iter):
        logging.info('Iteration %d/%d', it+1, n_iter)
        for iden in identifiers:
            if it > 0: model.decrement(iden)
            model.increment(iden)
        if it % 10 == 0:
            logging.info('Model: %s', model)
            ll = model.log_likelihood()
            ppl = math.exp(-ll / n_identifiers)
            logging.info('LL=%.0f ppl=%.3f', ll, ppl)

def train_model(identifiers, d, theta, char_lm_order=6):
    n_iter = 100
    char_lm = SRILMWrapper()
    char_lm.train(identifiers, char_lm_order, 'wbdiscount')
    base = CharLM(char_lm.ngram_file)
    model = PYP(d, theta, base)
    run_sampler(model, identifiers, n_iter)

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    train_data = '/home/pschulam/data/habeascorpus/datasets/train.txt'
    with open(train_data) as f:
        _, identifiers = corpus.read(f)

    train_model(identifiers, 0.5, 2)

if __name__ == '__main__':
    main()
