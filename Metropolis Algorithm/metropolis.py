from random import random, uniform
import numpy as np


def metropolis(X_0, N, delta, target, y, boundaries):
    markov_chain = np.zeros(N)
    markov_chain[0] = X_0
    for t in range(N - 1):
        X_t = markov_chain[t]
        Y = uniform(X_t - delta, X_t + delta)  # uniform proposal function
        if boundaries[0] <= Y <= boundaries[1]:
            w = target(Y, y)/target(X_t, y)
            alpha = min(w, 1)
            U = random()
            if U <= alpha:
                markov_chain[t + 1] = Y
            else:
                markov_chain[t + 1] = X_t
        else:
            markov_chain[t + 1] = X_t
    return markov_chain
