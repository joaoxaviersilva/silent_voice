import numpy as np


def entropy(p):
    v = np.array(list(p.values()))
    return -np.sum(v * np.log(v + 1e-6))


def reject(p, th=1.2):
    return entropy(p) > th
