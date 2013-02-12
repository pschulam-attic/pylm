import numpy as np

class Model:
    def __init__(self, log_likelihoods, weight=1.0):
        self._weight = weight
        self._probabilities = np.array([float(l) for l in log_likelihoods])
        self._weighted_probabilities = self._weight * self._probabilities

    def size(self):
        return self._probabilities.size

    def set_weight(self, weight):
        self._weight = weight
        self._weighted_probabilities = self._weight * self._probabilities

    def get_weight(self):
        return self._weight

    def get_probability_vector(self):
        return self._probabilities

    def get_weighted_probability_vector(self):
        return self._weighted_probabilities

class OptimizationProblem:
    def __init__(self, models, epsilon=0.0001):
        self._num_models = len(models)
        self._models = models
        self._size = self._models[0].size()
        weights = np.random.rand(self._num_models)
        weights /= np.sum(weights)
        self.update_weights(weights)
        self._converged = False
        self._epsilon = epsilon

    def converged(self):
        return self._converged

    def update_weights(self, weights):
        if not weights.size == self._num_models:
            raise Exception('{0} weights expected, received {1}'.format(self._num_models,
                                                                        weights.size))
        for w, m in zip(weights, self._models):
            m.set_weight(w)

    def update(self):
        old_ll = self.log_likelihood()

        weights = np.zeros(len(self._models))
        weighted_sum = self._weighted_sum()
        for i, m in enumerate(self._models):
            posterior = (np.sum(m.get_weighted_probability_vector() / weighted_sum)
                          / float(self._size))
            weights[i] = posterior
        self.update_weights(weights)

        new_ll = self.log_likelihood()

        if (new_ll - old_ll) / abs(new_ll) < self._epsilon:
            self._converged = True

    def log_likelihood(self):
        weighted_sum = self._weighted_sum()
        return np.sum(np.log(weighted_sum)) / float(self._size)

    def get_weight_vector(self):
        return [m.get_weight() for m in self._models]

    def _weighted_sum(self):
        weighted_sum = np.zeros(self._size)
        for m in self._models:
            weighted_sum += m.get_weighted_probability_vector()
        return weighted_sum

    def __repr__(self):
        ll = 'average log likelihood: {0}'.format(self.log_likelihood())
        ws = ['weight {0}: {1}'.format(i, m.get_weight())
              for i, m in enumerate(self._models)]
        return ll + '\n' + '\n'.join(ws)

def get_model_weights(*likelihood_lists):
    models = [Model(ll) for ll in likelihood_lists]
    optimization = OptimizationProblem(models)

    iterations = 0
    while not optimization.converged():
        iterations += 1
        optimization.update()

    return optimization.get_weight_vector()
