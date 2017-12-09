import numpy as np
# n is the number of rows
# m is the number of columns
def generateWeights (n, m):
	weights = np.random.rand(n, m)
	return weights