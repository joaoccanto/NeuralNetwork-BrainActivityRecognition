#from dataBreakDown import getTrainingData
from generateWeights import generateWeights
import numpy as np
from activationFunctions import sigmoid
from gradientFunctions import sigmoidPrime
from activationFunctions import relu
from activationFunctions import softMax
from gradientFunctions import reluPrime
from scipy import io



def backProp(label, output, hiddenLayerOutput, outputLayerInput, outputLayerWeights, hiddenLayerInput, data, hiddenLayerWeights):
	delta3 = - (label - output) * sigmoidPrime(outputLayerInput)
	hiddenLayerOutput.shape = (-1, len(hiddenLayerOutput))
	delta3.shape = (-1, len(delta3))
	dJdW2 = np.dot(hiddenLayerOutput.T, delta3)

	#dJdW2.shape = (len(dJdW2), -1)
	hiddenLayerInput.shape = (len(hiddenLayerInput), -1)

	delta2 = np.dot(outputLayerWeights, delta3.T)

	delta2 = delta2 * sigmoidPrime(hiddenLayerInput)
	# the arrays need to be reshaped into 2d arrays, to meet the .dot restrictions.
	data.shape = (-1, len(data)) 
	delta2.shape = (-1, len(delta2))
	dJdW1 = np.dot(data.T, delta2)

	return dJdW1, dJdW2


# data --> the data being used for the NN training
# layerStructure --> an array of 3 ints; each describes number of perceptrons per layer.
def trainingNN(data, layerStructure):
	#generate the weights for each layer
	hiddenLayerWeights = generateWeights(layerStructure[0] + 1, layerStructure[1])
	outputLayerWeights = generateWeights(layerStructure[1], layerStructure[2])
	label = [1, 0]
	bias = 1
	learningRate = .1
	file = open("Run3.txt", "w");
	
	for x in range(0, len(data)):
		data[x] = data[x]
		
		data[x] = np.hstack([data[x], np.ones(1)])
		
		#summation
		hiddenLayerInput = np.dot(data[x], hiddenLayerWeights)

		#pass first layer output through activation function
		hiddenLayerOutput = sigmoid(hiddenLayerInput)

		#hiddenLayerOutput = np.hstack([hiddenLayerOutput, np.ones(1)])
		#summation
		outputLayerInput = np.dot(hiddenLayerOutput, outputLayerWeights)

		
		#pass the input through the activation function
		output = sigmoid(outputLayerInput)
		
		error = (label - output) ** 2
		#gradient descent

		dJdW1, dJdW2 = backProp(label, output, hiddenLayerOutput, outputLayerInput, outputLayerWeights, hiddenLayerInput, data[x], hiddenLayerWeights)

		# here is where the error is at
		#update the weights
		hiddenLayerWeights = hiddenLayerWeights +  (dJdW1 * learningRate)
		outputLayerWeights = outputLayerWeights + (dJdW2 * learningRate)
		
		print("training ... ")

		file.write("x ==>> %d" % x)
		file.write("\t error1 ==> %f" % error[0])
		file.write("\t error2 ==> %f" % error[1])
		file.write("\t output1 => %f" % output[0])
		file.write("\t output2 => %f \n" % output[1])

		
	file.close()
	return hiddenLayerWeights, outputLayerWeights


##########################################################################
def backPropRelu(label, output, hiddenLayerOutput, outputLayerInput, outputLayerWeights, hiddenLayerInput, data, hiddenLayerWeights):
	
	delta3 = - (label - output) * reluPrime(outputLayerInput, True)
	hiddenLayerOutput.shape = (-1, len(hiddenLayerOutput))
	delta3.shape = (-1, len(delta3))
	dJdW2 = np.dot(hiddenLayerOutput.T, delta3)

	#dJdW2.shape = (len(dJdW2), -1)
	hiddenLayerInput.shape = (len(hiddenLayerInput), -1)

	delta2 = np.dot(outputLayerWeights, delta3.T)

	delta2 = delta2 * reluPrime(hiddenLayerInput)
	# the arrays need to be reshaped into 2d arrays, to meet the .dot restrictions.
	data.shape = (-1, len(data)) 
	delta2.shape = (-1, len(delta2))
	dJdW1 = np.dot(data.T, delta2)

	return dJdW1, dJdW2


def trainingWithRelu(data, layerStructure):
	#generate the weights for each layer
	hiddenLayerWeights = generateWeights(layerStructure[0], layerStructure[1])
	outputLayerWeights = generateWeights(layerStructure[1], layerStructure[2])
	label = 1
	bias = 1
	learningRate = .3
	file = open("reluTrainng.txt", "w");
	
	for x in range(len(data)):
		
		#summation
		hiddenLayerInput = np.dot(data[x], hiddenLayerWeights)

		#pass first layer output through activation function
		hiddenLayerOutput = relu(hiddenLayerInput)

		#hiddenLayerOutput = np.hstack([hiddenLayerOutput, np.ones(1)])
		#summation
		outputLayerInput = np.dot(hiddenLayerOutput, outputLayerWeights)

		
		#pass the input through the activation function
		output = softMax(outputLayerInput)

		error = (label - output) ** 2
		#gradient descent

		dJdW1, dJdW2 = backPropRelu(label, output, hiddenLayerOutput, outputLayerInput, outputLayerWeights, hiddenLayerInput, data[x], hiddenLayerWeights)

		# here is where the error is at
		#update the weights
		hiddenLayerWeights = hiddenLayerWeights -  (dJdW1 * learningRate)
		outputLayerWeights = outputLayerWeights - (dJdW2 * learningRate)
		
		print("processing ... ")

		#file.write("x ==>> %d" % x)
		#file.write("\t error ==> %f" % error)
		#file.write("\t output => %f \n" % output)
		
	file.close()
	return hiddenLayerWeights, outputLayerWeights