from activationFunctions import sigmoid

def sigmoidPrime(z):
    return sigmoid(z) * (1 - sigmoid(z))
def reluPrime(z):
	for x in range (0, len(z)):
			if z[0][x]  > 0:
				z[0][x] = 1
			else:
				z[0][x] = 0
	return z