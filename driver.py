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
#io.savemat('weights', dict(hiddenLayerWeights=hiddenLayerWeights, outputLayerWeights=outputLayerWeights))

'''
mat = io.loadmat("S02.mat")
struct1 = mat["data"]
data1 = struct1[0][0]["X"][0][0]
data1New = data1.astype(type('float', (float,), {}))

mat = io.loadmat("S02.mat")
struct2 = mat["data"]
data2 = struct2[0][1]["X"][0][0]
data2New = data2.astype(type('float', (float,), {}))

mat = io.loadmat("S02.mat")
struct3 = mat["data"]
data3 = struct3[0][2]["X"][0][0]
data3New = data3.astype(type('float', (float,), {}))

mat = io.loadmat("S02.mat")
struct4 = mat["data"]
data4 = struct4[0][3]["X"][0][0]
data4New = data4.astype(type('float', (float,), {}))


chart1 = leather.Chart('Task One')
chart1.add_dots(data1New)
chart1.to_svg('taskOne2.svg')

chart2 = leather.Chart('Task Two')
chart2.add_dots(data2New)
chart2.to_svg('taskTwo2.svg')

chart3 = leather.Chart('Task Three')
chart3.add_dots(data3New)
chart3.to_svg('taskThree2.svg')
 
chart4 = leather.Chart('Task Four')
chart4.add_dots(data4New)
chart4.to_svg('taskFour2.svg')

'''