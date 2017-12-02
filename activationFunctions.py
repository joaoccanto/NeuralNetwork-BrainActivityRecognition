import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp (-z))
def tanh(z):
    return (2 / (1 + np.exp(-2 * z))) - 1
def ReLu(z):
    if z < 0:
        return 0
    else:
        return 1
