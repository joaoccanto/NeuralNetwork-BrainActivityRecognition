
import numpy as np
from activationFunctions import sigmoid
from scipy import io

def processTestData(data, hiddenLayerWeights, outputLayerWeights, fileName):
	label = 1
	bias = 1
	file = open(fileName, "w");
	meanOutput = 0
	
	for x in range(0, len(data)):
	
		#convert data from milliseconds to seconds
		data[x] = data[x] / 1000
	
		#summation
		hiddenLayerInput = np.dot(data[x], hiddenLayerWeights)

		#pass first layer output through activation function
		hiddenLayerOutput = sigmoid(hiddenLayerInput + bias)

		#summation
		outputLayerInput = np.dot(hiddenLayerOutput, outputLayerWeights)

		#pass the input through the activation function
		output = sigmoid(outputLayerInput + bias)
		meanOutput = meanOutput + output
		
		print("processing ... ")

		file.write("x ==>> %d" % x)
		file.write("\t output => %f \n" % output)

	meanOutput = meanOutput / len(data)

	file.write("meanOutput ===> % f" % meanOutput)
		
	file.close()
	return hiddenLayerWeights, outputLayerWeights