import numpy as np

# return an percentage
def accuracy(y_expect, y_pred):
    assert y_expect.shape == y_pred.shape
    return np.sum(y_expect == y_pred) / y_pred.shape[0] * 100
