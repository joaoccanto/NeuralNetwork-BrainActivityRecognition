import numpy as np

def sigmoid(z):
	z = np.clip( z, -500, 500 )
	z = 1 / (1 + np.exp (-z))
	return z

def tanh(z):
    return (2 / (1 + np.exp(-2 * z))) - 1
def relu(z):
	for x in range (0, len(z[0])):
			if z[0][x]  > 0:
				pass
			else:
				z[0][x] = 0
	return z
def softMax (z):
	exps = np.exp(z - np.max(z))
	return exps / np.sum(exps)

