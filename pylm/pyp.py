import math
import random

class CRP(object):
    def __init__(self):
        self.tables = {}
        self.ntables = 0
        self.ncustomers = {}
        self.total_customers = 0

    def _seat_to(self, k, i):
        if not k in self.tables: # add new dish
            self.tables[k] = []
            self.ncustomers[k] = 0
        self.ncustomers[k] += 1
        self.total_customers += 1
        tables = self.tables[k]
        if i == -1: # add new table
            self.ntables += 1
            tables.append(1)
        else: # existing table
            tables[i] += 1
        return (i == -1)

    def _unseat_from(self, k, i):
        self.ncustomers[k] -= 1
        self.total_customers -= 1
        tables = self.tables[k]
        tables[i] -= 1
        if tables[i] == 0: # cleanup empty table
            del tables[i]
            self.ntables -= 1
            if len(tables) == 0:
                del self.tables[k]
                del self.ncustomers[k]
            return True
        return False

class PYP(CRP):
    def __init__(self, d, theta, base):
        super(PYP, self).__init__()
        self.discount = float(d)
        self.strength = float(theta)
        self.base = base

    @property
    def d(self):
        return self.discount

    @property
    def theta(self):
        return self.strength

    def _sample_table(self, k):
        if k not in self.tables: return -1
        p_new = (self.theta + self.d * self.ntables) * self.base.prob(k)
        norm = p_new + self.ncustomers[k] - self.d * len(self.tables[k])
        x = random.random() * norm
        for i, c in enumerate(self.tables[k]):
            if x < c - self.d: return i
            x -= c - self.d
        return -1

    def _customer_table(self, k, n):
        tables = self.tables[k]
        for i, c in enumerate(tables):
            if n < c: return i
            n -= c

    def increment(self, k):
        i = self._sample_table(k)
        if self._seat_to(k, i):
            self.base.increment(k)

    def decrement(self, k):
        i = self._customer_table(k, random.randrange(0, self.ncustomers[k]))
        if self._unseat_from(k, i):
            self.base.decrement(k)

    def prob(self, k):
        # new table
        w = (self.theta + self.d * self.ntables) * self.base.prob(k)
        # existing tables
        if k in self.tables:
            w += self.ncustomers[k] - self.d * len(self.tables[k])
        return w / (self.theta + self.total_customers)

    def log_likelihood(self):
        if self.d == 0:
            ll = (math.lgamma(self.theta) - math.lgamma(self.theta + self.total_customers)
                  + self.ntables * math.log(self.theta))
        else:
            ll = (math.lgamma(self.theta) - math.lgamma(self.theta + self.total_customers)
                  + math.lgamma(self.theta / self.d + self.ntables)
                  - math.lgamma(self.theta / self.d)
                  + self.ntables * (math.log(self.d) - math.lgamma(1 - self.d))
                  + sum(math.lgamma(c - self.d) for tables in self.tables.itervalues()
                        for c in tables))
        return ll

    def __repr__(self):
        return 'PYP(d={self.d}, theta={self.theta})'.format(self=self)
