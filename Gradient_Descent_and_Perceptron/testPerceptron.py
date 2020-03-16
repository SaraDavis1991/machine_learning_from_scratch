###################################################################################################################################################
# filename: testPreceptron.py
# author: Sara Davis 
# date: 10/10/2018
# version: 1.0
# description: runs testPerceptron.py
#############################################################################################################################################

import numpy as np

from perceptron import perceptron_train
from perceptron import perceptron_test

#X_test = np.array([[0,1], [1,0], [5,4], [1,1], [3,3], [2,4], [1,6]]) #should be -2, -2, 6
#Y_test = np.array([[1], [1], [0], [1], [0], [0], [0]])
X_test = np.array([[0,0], [1,1], [0,1],[2,2],[1,0],[1,2]])
Y_test = np.array([[-1],[1],[-1],[1],[-1], [1]])
X = np.array([[-2, 1], [1,1], [1.5, -.5], [-2, -1], [-1, -1.5], [2, -2]])
Y = np.array([[1], [1], [1], [-1], [-1], [-1]])

W = perceptron_train(X, Y)
print(W)




test_acc = perceptron_test(X_test, Y_test, W[0], W[1])
print(test_acc)
