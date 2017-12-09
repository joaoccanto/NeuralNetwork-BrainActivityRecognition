from scipy import io
from math import ceil
from math import floor
import numpy as np



def getTrainingdData(dataFile, percentage):
	mat = io.loadmat(dataFile)
	data = mat['datainmicrovolts']
	numOfRows = len(data)
	numOfTrainingRows = int (ceil(numOfRows * percentage))
	trainingData = []
	for x in range (0, numOfTrainingRows):
		trainingData.append(data[x])
	return trainingData

def getTestData(dataFile, percentage):
	mat = io.loadmat(dataFile)
	data = mat['datainmicrovolts']
	numOfRows = len(data)
	numOfTestRows = int (floor(numOfRows * (1 -percentage)))
	testData = []
	for x in range (numOfTestRows + 1, numOfRows):
		testData.append(data[x])
	return testData

def getTrainingDataByPins(dataFile, percentage, pins):
	mat = io.loadmat(dataFile)
	data = mat['datainmicrovolts']
	numOfRows = len(data)
	numOfTestRows = int (ceil(numOfRows * percentage))
	rowData = np.arange(len(pins))
	testData = []
	for x in range (0, numOfTestRows):
		for y in range (0, len(pins)):
			rowData[y] = data[x][y]
		testData.append(rowData)
	return testData

def getTestDataByPins(dataFile, percentage, pins):
	mat = io.loadmat(dataFile)
	data = mat['datainmicrovolts']
	numOfRows = len(data)
	numOfTestRows = int (floor(numOfRows * (1 -percentage)))
	rowData = np.arange(len(pins))
	testData = []
	for x in range (numOfTestRows + 1, numOfRows):
		for y in range (0, len(pins)):
			rowData[y] = data[x][y]
		testData.append(rowData)
	return testData
