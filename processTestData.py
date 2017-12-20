
import numpy as np
from activationFunctions import relu
from activationFunctions import softMax
from activationFunctions import sigmoid
from scipy import io

def processTestData(data, hiddenLayerWeights, outputLayerWeights, fileName):
	label = 1
	bias = 1
	file = open(fileName, "w");
	meanOutput = 0
	
	for x in range(0, len(data)):
	
		#data[x] = np.hstack([data[x], np.ones(1)])

	
		#summation
		hiddenLayerInput = np.dot(data[x], hiddenLayerWeights )

		#pass first layer output through activation function
		hiddenLayerOutput = sigmoid(hiddenLayerInput + 1)

		#summation
		outputLayerInput = np.dot(hiddenLayerOutput, outputLayerWeights)

		#pass the input through the activation function
		output = sigmoid(outputLayerInput + 1)
		meanOutput = meanOutput + output
		
		print("processing ... ")

		file.write("x ==>> %d" % x)
		file.write("\t output1 => %f" % output[0])
		file.write("\t output2 => %f \n" % output[1])

	meanOutput1 = meanOutput[0] / len(data)
	meanOutput2 = meanOutput[1] / len(data)

	file.write("meanOutput1 ===> %f " % meanOutput1)
	file.write("meanOutput2 ===> %f " % meanOutput2)
		
	file.close()
	return meanOutput1, meanOutput2