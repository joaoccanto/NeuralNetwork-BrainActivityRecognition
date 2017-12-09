from activationFunctions import sigmoid
import numpy as np
from scipy import io
import leather
from generateWeights import generateWeights
import dataBreakDown
from trainingNN import trainingNN
from processTestData import processTestData

#mat = io.loadmat('011-2015-1.mat')
#data = mat['datainmicrovolts']
#print(data.shape)

trainingData = dataBreakDown.getTrainingDataByPins('011-2015-1.mat', .01, [74, 75])
testData1 = dataBreakDown.getTestDataByPins('011-2015-1.mat', .30, [74, 75])
testData2 = dataBreakDown.getTestDataByPins('011-2015-2.mat', .30, [74, 75])

hiddenLayerWeights, outputLayerWeights = trainingNN(trainingData, [len(trainingData[0]), 3, 1])

processTestData(testData1, hiddenLayerWeights, outputLayerWeights, "TestDataFromSamePool2")
processTestData(testData2, hiddenLayerWeights, outputLayerWeights, "TestDataFromDifferentPool2")

