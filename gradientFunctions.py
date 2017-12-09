from activationFunctions import sigmoid

def sigmoidPrime(z):
    return sigmoid(z) * (1 - sigmoid(z))
