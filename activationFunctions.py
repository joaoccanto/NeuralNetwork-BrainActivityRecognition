import math

def sigmoid(z):
    return 1 / (1 + math.e ** -z)
def tanh(z):
    return (2 / (1 + math.e ** (-2 * z))) - 1
def ReLu(z):
    if z < 0:
        return 0
    else:
        return 1
