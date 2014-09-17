# -*- coding: utf-8 -*-
__author__ = 'Cminx'

import numpy as np
import math


class Perceptron:
    '''
	    参数X和Lable都为Numpy Array对象，其中X为Data矩阵，Lable为标记向量
		:param threshold 收敛阀值
		:param rho     learning rate(学习率)
		:param maxStep   最大迭代步数
	'''

    def __init__(self, threshold=0.001, rho=0.01, maxStep=1000):
        self.threshold = threshold
        self.rho = rho
        self.maxStep = maxStep

    '''
	初始化变量，将形成一个theta矩阵，包括W向量以及b向量
	'''

    def initializeParameters(self, paraCount):
        W = np.random.random(paraCount)
        b = np.zeros(1, dtype='float64')
        self.theta = np.hstack((W, b))

    '''
		损失函数
		函数主要确定了模型需要优化的损失函数：
		costFunc = -sum(y * lable) = -sum((W*X+b) * lable) 目标最小化损失函数
		采用梯度下降，这里使用最速下降最小化损失函数来求出参数theta(W,b)
		Wnew = W - deltaW = W - rho * costFunc对W的偏导数 = W - rho * (-X * lable) = W + rho * X * lable
		bNew = b - deltab = b - rho * costFunction对b的偏导数 = W + rho * lable
		由于使用最速下降，所以每次都将迭代所有数据确定梯度grand,等于所有错误数据改变梯度总量的平均值
		deltaW = sum(rho * Xi * lable)/errorCount
		deltab = sum(rho * lable)/errorCount
		其中rho为学习率
		该方法返回的值为
		(cost, theta)  即损失值以及参数改变值
	'''

    def perceptronCost(self, X, Lable):
        cost = 0.0
        W = self.theta[:X.shape[1]]
        b = self.theta[X.shape[1]:]
        deltaW = np.zeros(shape=W.shape)
        deltab = np.zeros(shape=b.shape)
        errorCount = 0
        for i in range(X.shape[0]):
            y = 0 if (W.dot(X[i]) + b) * Lable[i] > 0 else 1
            cost += y
            if y > 0:
                errorCount += 1
                deltaW += self.rho * Lable[i] * X[i]
                deltab += self.rho * Lable[i]
        if errorCount > 0:
            deltaW /= errorCount
            deltab /= errorCount
        return cost, np.hstack((deltaW, deltab))

    '''
		最优化参数，通过costFunc确定的cost来判断是否收敛，如果错误分类数小于某个阀值则认为收敛。
	'''

    def min(self, X, Lable):
        cost, delta = self.perceptronCost(X, Lable)
        step = 0
        while cost > self.threshold and step < self.maxStep:
            step += 1
            self.theta[:X.shape[1]] += self.rho * delta[:X.shape[1]]
            self.theta[X.shape[1]:] += self.rho * delta[X.shape[1]:]
            cost, delta = self.perceptronCost(X, Lable)
            print 'Gradient descent step %d, residual is %f' % (step, cost)


    def train(self, X, Lable):
        self.initializeParameters(X.shape[1])
        self.min(X, Lable)

    def classify(self, X):
        if self.theta.dot(np.hstack((X, np.ones(1)))) > 0:
            return 1
        else:
            return -1