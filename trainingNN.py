#from dataBreakDown import getTrainingData
from generateWeights import generateWeights
import numpy as np
from activationFunctions import sigmoid
from gradientFunctions import sigmoidPrime
from scipy import io




def gradientDescent(label, output, hiddenLayerOutput, outputLayerInput, outputLayerWeights, hiddenLayerInput, data, hiddenLayerWeights):
	delta3 = - (label - output) * sigmoidPrime(outputLayerInput)
	hiddenLayerOutput.shape = (-1, len(hiddenLayerOutput))
	#delta3.shape = (-1, len(delta3))
	dJdW2 = np.dot(hiddenLayerOutput.T, delta3)

	dJdW2.shape = (len(dJdW2), -1)
	delta2 = np.dot(outputLayerWeights, delta3) * sigmoidPrime(hiddenLayerInput)
	# the arrays need to be reshaped into 2d arrays, to meet the .dot restrictions.
	data.shape = (-1, len(data)) 
	delta2.shape = (-1, len(delta2))
	dJdW1 = np.dot(data.T, delta2)

	return dJdW1, dJdW2


# data --> the data being used for the NN training
# layerStructure --> an array of 3 ints; each describes number of perceptrons per layer.
def trainingNN(data, layerStructure):
	#generate the weights for each layer
	hiddenLayerWeights = generateWeights(layerStructure[0], layerStructure[1])
	outputLayerWeights = generateWeights(layerStructure[1], layerStructure[2])
	label = 1
	bias = 1
	learningRate = .2
	file = open("Run3.txt", "w");
	
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

		error = (label - output) ** 2
		#gradient descent

		dJdW1, dJdW2 = gradientDescent(label, output, hiddenLayerOutput, outputLayerInput, outputLayerWeights, hiddenLayerInput, data[x], hiddenLayerWeights)

		# here is where the error is at
		#update the weights
		hiddenLayerWeights = hiddenLayerWeights -  (dJdW1 * learningRate)
		outputLayerWeights = outputLayerWeights - (dJdW2 * learningRate)
		
		print("processing ... ")

		file.write("x ==>> %d" % x)
		file.write("\t error ==> %f" % error)
		file.write("\t output => %f \n" % output)
		
	file.close()
	return hiddenLayerWeights, outputLayerWeights
	#io.savemat("./hiddenLayerWeights.mat", mdict={"arr" : arr})
	#io.savemat("./outputLayerWeights.mat", mdict={"arr" : arr})

