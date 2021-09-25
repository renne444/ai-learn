import numpy as np


def demean(X):
    return X - np.mean(X)


def direction(w):
    return w / np.linalg.norm(w)
