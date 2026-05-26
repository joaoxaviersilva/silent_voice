import numpy as np


def normalize(p):
    c = np.mean(p, axis=0)
    p = p - c
    s = np.max(np.linalg.norm(p, axis=1)) + 1e-6
    return p / s


def extract_sequence(seq):
    feats = []
    prev = None

    for p in seq:
        p = normalize(p)

        v = np.linalg.norm(p[0] - p[10])
        h = np.linalg.norm(p[5] - p[15]) + 1e-6

        mar = v / h
        g = np.array([mar, v, h])

        vel = g - prev if prev is not None else np.zeros_like(g)

        feats.append(np.concatenate([g, vel]))
        prev = g

    return np.array(feats)
