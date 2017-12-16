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


file = open("taskFour/outputs.txt", "w");
file.write("Task Four\n\n")
for x in range(0, 10):
	trainingData = dataBreakDown.getTrainingData("S02.mat", .999, 'data', 'X', 3)
	testData1 = dataBreakDown.getTestData("S02.mat", .001, 'data', 'X', 0)
	testData2 = dataBreakDown.getTestData("S02.mat", .001, 'data', 'X', 1)
	testData3 = dataBreakDown.getTestData("S02.mat", .001, 'data', 'X', 2)
	testData4 = dataBreakDown.getTestData("S02.mat", .001, 'data', 'X', 3)

	hiddenLayerWeights, outputLayerWeights = trainingNN(trainingData, [len(trainingData[0]),1, 2])

	m1, m2 = processTestData(testData1, hiddenLayerWeights, outputLayerWeights, "taskFour/TaskFour ===> TaskOne(" + str(x) + ")")
	file.write("Task One: \t %f" % m1)
	file.write("\t %f\n" % m2)
	m1, m2 = processTestData(testData2, hiddenLayerWeights, outputLayerWeights, "taskFour/TaskFour ===> TaskTwo(" + str(x) + ")")
	file.write("Task Two: \t %f" % m1)
	file.write("\t %f\n" % m2)
	m1, m2 = processTestData(testData3, hiddenLayerWeights, outputLayerWeights, "taskFour/TaskFour ===> TaskThree(" + str(x) + ")")
	file.write("Task Three: \t %f" % m1)
	file.write("\t %f\n" % m2)
	m1, m2 = processTestData(testData4, hiddenLayerWeights, outputLayerWeights, "taskFour/TaskFour ===> TaskFour(" + str(x) + ")")
	file.write("Task Four: \t %f" % m1)
	file.write("\t %f\n\n\n" % m2)

file.close
io.savemat('weights', dict(hiddenLayerWeights=hiddenLayerWeights, outputLayerWeights=outputLayerWeights))
