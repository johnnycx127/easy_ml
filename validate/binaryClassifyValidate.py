__author__ = 'Cminx'

import numpy as np


class BiClassifyValidater:
    def __init__(self, model, lable=(1, -1)):
        self.classifier = model
        self.confusion = np.zeros((2, 2))
        self.positive = lable[0]
        self.negative = lable[1]


    def buildConfusion(self, X, Lable):
        for i in range(X.shape[0]):
            h = self.classifier.classify(X[i])
            if self.positive == Lable[i] and self.positive == h:
                self.confusion[0][0] += 1
            if self.negative == Lable[i] and self.positive == h:
                self.confusion[0][1] += 1
            if self.positive == Lable[i] and self.negative == h:
                self.confusion[1][0] += 1
            if self.negative == Lable[i] and self.negative == h:
                self.confusion[1][1] += 1
        print self.confusion

    def getAccuracy(self):
        accuracy = (self.confusion[0][0] + self.confusion[1][1]) / self.confusion.sum()
        return accuracy

    def getPisitionAndRecall(self):
        prisition = self.confusion[0][0] / self.confusion[0].sum()
        recall = self.confusion[0][0] / self.confusion.T[0].sum()
        return prisition, recall

