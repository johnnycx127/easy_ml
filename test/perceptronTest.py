# -*- coding: utf-8 -*-
__author__ = 'Cminx'

import numpy as np
import pylab as plt
import random, os
from easy_ml.classifier.perceptron import Perceptron
from easy_ml.validate.binaryClassifyValidate import BiClassifyValidater

dataSet = np.zeros((149,2))
lables = np.zeros(149)

trainX = np.zeros((100, 2))
trainLable = np.zeros(100)

testX = np.zeros((49, 2))
testLable = np.zeros(49)

def convertDate(path):
	global lables,dataSet
	f = file(path,'r')
	index = 0
	while True:
		line = f.readline()
		if not line :
			break
		items = line.replace('\n','').split('\t')
		lables[index] = items[0]
		dataSet[index][:] = items[1:]
		index += 1
	f.close()

def sampleTrainAndTest():
	global trainX, trainLable, testX, testLable
	trainX = dataSet[:100]
	trainLable = lables[:100]
	testX = dataSet[100 : ]
	testLable = lables[100 : ]

def createData(path):
	dataFile = file(path, 'w')
	w = 0.8
	b = 0.3
	for x in range(1,150,1):
		y = w * x + b
		if x % 2 == 0:
			y += random.random() * 5
			c = 1
		else :
			y -= random.random() * 6
			c = -1
		dataFile.write(str(c) + '\t' + str(y) + '\t' + str(x))
		dataFile.write('\n')
	dataFile.close()

if __name__ == '__main__':
	dataDir = '../../data'
	dataFileName = 'linear_perceptron.data'
	dataPath = os.path.join(dataDir, dataFileName)
	if dataFileName not in os.listdir(dataDir):
		createData(dataPath)
	convertDate(dataPath)
	sampleTrainAndTest()
	model = Perceptron(threshold=1,rho=0.1, maxStep=10000)
	model.train(trainX, trainLable)
	print model.theta
	#validate
	validater = BiClassifyValidater(model)
	validater.buildConfusion(testX, testLable)
	print 'model accuracy is %f' %validater.getAccuracy()
	print 'model prisition is %f, recall is %f' %(validater.getPisitionAndRecall())
	#draw
	starPoints = ([],[])
	plusPoints = ([],[])

	for i in range(len(lables)):
		if lables[i] > 0:
			starPoints[0].append(dataSet[i][0])
			starPoints[1].append(dataSet[i][1])
		else:
			plusPoints[0].append(dataSet[i][0])
			plusPoints[1].append(dataSet[i][1])
	plt.plot(starPoints[0], starPoints[1], '*')
	plt.plot(plusPoints[0], plusPoints[1], '+')
	plt.plot([0 ,testX.T[0].max() + 1],[-1 * model.theta[2] / model.theta[1],-1 * model.theta[0] * (testX.T[0].max()+1) / model.theta[1]], '-', lw=1)
	plt.show()
