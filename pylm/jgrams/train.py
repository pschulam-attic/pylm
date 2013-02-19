import argparse
import logging
import math

from ..pyp import PYP
from ..arpa import CharLM, SRILMWrapper

import corpus

def run_sampler(model, identifiers, n_iter):
    n_identifiers = len(identifiers)
    for it in range(n_iter):
        logging.info('Iteration %d/%d', it, n_iter)
        for iden in identifiers:
            if it > 0: model.decrement(iden)
            model.increment(iden)
        #if it % 10 == 0:
        logging.info('Model: %s', model)
        ll = model.log_likelihood()
        ppl = math.exp(-ll / n_identifiers)
        logging.info('LL=%.0f ppl=%.3f', ll, ppl)

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--traindata', help='path to training data', required=True)
    parser.add_argument('--discount', help='discount parameter for PYP', required=True, type=float)
    parser.add_argument('--strength', help='strength parameter for PYP', required=True, type=float)
    parser.add_argument('--niter', help='number of iterations of sampling', type=int, default=10)
    parser.add_argument('--char_lm_order', help='order of character language model', type=int, default=10)
    args = parser.parse_args()

    with open(args.traindata) as f:
        _, identifiers = corpus.read(f)

    char_lm = SRILMWrapper()
    char_lm.train(identifiers, args.char_lm_order, 'wbdiscount')
    base = CharLM(char_lm.ngram_file)
    assert args.strength > - args.discount
    model = PYP(args.discount, args.strength, base)
    run_sampler(model, identifiers, args.niter)

if __name__ == '__main__':
    main()
