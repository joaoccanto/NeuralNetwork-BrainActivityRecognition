from activationFunctions import sigmoid
import numpy as np
from scipy import io
import leather
from generateWeights import generateWeights
import dataBreakDown
from trainingNN import trainingNN
from trainingNN import trainingWithRelu
from processTestData import processTestData
from graphData import dotGraph

#mat = io.loadmat('011-2015-1.mat')
#data = mat['datainmicrovolts']
#print(data.shape)

#trainingData = dataBreakDown.getTrainingDataByPins('011-2015-1.mat', .01, [74, 75])
#testData1 = dataBreakDown.getTestDataByPins('011-2015-1.mat', .30, [74, 75])
#testData2 = dataBreakDown.getTestDataByPins('011-2015-2.mat', .30, [74, 75])

#trainingData = dataBreakDown.getTrainingData('011-2015-1.mat', .2, 'key1', 'key2', 1)
#testData1 = dataBreakDown.getTestData('011-2015-1.mat', .30, 'key1', 'key2', 1)
#testData2 = dataBreakDown.getTestData('011-2015-2.mat', .30, 'key1', 'key2', 1)

trainingData = dataBreakDown.getTrainingData("S02.mat", .1, 'data', 'X', 0)
testData1 = dataBreakDown.getTestData("S02.mat", .3, 'data', 'X', 0)
testData2 = dataBreakDown.getTestData("S01.mat", .3, 'data', 'X', 0)


hiddenLayerWeights, outputLayerWeights = trainingNN(trainingData, [len(trainingData[0]), 5, 1])

processTestData(testData1, hiddenLayerWeights, outputLayerWeights, "TestDataFromSamePool2")
processTestData(testData2, hiddenLayerWeights, outputLayerWeights, "TestDataFromDifferentPool2")

'''
mat = io.loadmat("S01.mat")
struct1 = mat["data"]
data1 = struct1[0][0]["X"][0][0]
data1New = data1.astype(type('float', (float,), {}))

mat = io.loadmat("S02")
struct2 = mat["data"]
data2 = struct2[0][0]["X"][0][0]
data2New = data2.astype(type('float', (float,), {}))


print(data1New.shape)
print(data2.shape)
#dotGraph("data1.svg", data1New)
dotGraph("data2.svg", data2New)

'''