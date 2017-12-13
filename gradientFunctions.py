from activationFunctions import sigmoid

def sigmoidPrime(z):
    return sigmoid(z) * (1 - sigmoid(z))
def reluPrime(z, flag=False):
	if flag == False:
		for x in range (0, len(z[0])):
			if z[0][x]  > 0:
				z[0][x] = 1
			else:
				z[0][x] = 0
	else:
		for x in range (0, len(z)):
			if z[x]  > 0:
				z[x] = 1
			else:
				z[x] = 0
	return z

