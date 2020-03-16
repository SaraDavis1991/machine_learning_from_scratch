###################################################################################################################################################
# filename: nn.py
# author: Sara Davis 
# date: 12/3/2018
# version: 2.0
# description: Create a simple neural network from scratch
###################################################################################################################################################

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_moons
import random

###########################################################################################################
# def calculate_loss(model, X, y)
# This function calculates the loss
# inputs: model, X, y
# returns: totalLoss
############################################################################################################
#calculate the loss
def calculate_loss(model, X, y):
	W1 = model['W1']
	W2 = model['W2']
	b1 = model['b1']
	b2 = model['b2']

	a = X.dot(W1) + b1 
	h = np.tanh(a)
	z = h.dot(W2) + b2
	raiseToe = np.exp(z) #raise to e
	total = np.sum(raiseToe, axis = 1, keepdims = True) #calculate the sum of values raised to e power
	softmax = raiseToe /total #calculate the softmax values for all points


	N = len(X)
	sample = - 1/N #sample size
	findLargest = softmax[range(N), y]
	singlePtLosses = np.log(findLargest) #find the worst loss from the softmax activation
	overallLoss = np.sum(singlePtLosses)
	finalLoss = sample * overallLoss

	return finalLoss
###########################################################################################################
# def predict(model, x)
# This makes a prediction of label for each test datapoint
# inputs: model, x
# returns: largest (the most likely class that each data point belongs to)
############################################################################################################
#calculate the output of the network using forward propagation, return highest prob
def predict(model, x):
	W1 = model['W1']
	W2 = model['W2']
	b1 = model['b1']
	b2 = model['b2']

	a = x.dot(W1) + b1 #use the weights and biases learned from model to do dot product with test point
	h = np.tanh(a)
	z = h.dot(W2) + b2
	
	raiseToe = np.exp(z)
	total = np.sum(raiseToe, axis = 1, keepdims = True)
	softmax = raiseToe / total #calculate softmax value at node

	largest = np.argmax(softmax, axis = 1) #find most likely class

	return largest

###########################################################################################################
# def build_model(X, y, nn_hdim, num_passes = 20000, print_loss = False)
# This does backpropagation for the network (2 in, 2 out) to learn the W and B, return the model containing the W and B
# inputs: X, y, nn_hdim, num_passes = 20000, print_loss = False
# returns: the built neural network
############################################################################################################
def build_model(X, y, nn_hdim, num_passes = 20000, print_loss = False):
	
	np.random.seed(0) #seed to randomly initialize weights
	eta = 0.01

	W1 = np.random.randn(2, nn_hdim)#  populating an array of weights for each input to each node
	W2 = np.random.randn(nn_hdim, 2) # this is populating  an array of weights for each output to each node
	b1 = np.zeros((1, nn_hdim)) #create an array with 0's the length of the input
	b2 = np.zeros((1, 2))#create an array with 0's the length of the output
	model = {}
	for i in range(num_passes): #loop until w and b are minimized
		a = X.dot(W1) + b1 
		h = np.tanh(a)
		z = h.dot(W2) + b2


		raiseToe = np.exp(z)
		total = np.sum(raiseToe, axis =1, keepdims = True)
		softmax = raiseToe / total #calculate softmax value at node


		softmax[range(len(X)), y] -= 1 #decrement layer
		

		PartialW2 = np.transpose(h) #Transpose so that the we multiply each node by its matching weight. Does not work without
		PartialW2 = PartialW2.dot(softmax)
		BiasLayer2 = np.sum(softmax, axis = 0, keepdims = True)
		

		loss = 1 - np.power(h, 2)
		WT2 = np.transpose(W2) #Transpose so that the we multiply each node by its matching weight. Does not work without
		propagatedLoss = softmax.dot(WT2) * loss
		
		WT1 = np.transpose(X) #Transpose so that we multiply each node by its matching weight. Does not work without
		PartialW1 = np.dot(WT1, propagatedLoss)
		BiasLayer1 = np.sum(propagatedLoss, axis = 0)


		W1 = W1 - (eta* PartialW1) #decrement weights and biases as you learn until loss is minimized
		W2 = W2 - (eta* PartialW2)
		b1 = b1 - (eta* BiasLayer1)
		b2 = b2 - (eta* BiasLayer2)

		model = {'W1': W1, 'b1': b1, 'W2' : W2, 'b2':b2} #build the model

		if print_loss == True and num_passes % 1000 == 0:
			print(calculate_loss(model, X, y))

	return model




###########################################################################################################
# def build_model(X, y, nn_hdim, num_passes = 20000, print_loss = False)
# This does backpropagation for the network (2 in, 3 out) to learn the W and B, return the model containing the W and B
# inputs: X, y, nn_hdim, num_passes=20000, print_loss = False
# returns: the built neural network
############################################################################################################
def build_model_691(X, y, nn_hdim, num_passes=20000, print_loss = False):
	np.random.seed(0) #seed to randomly initialize weights
	eta = 0.01

	W1 = np.random.randn(2, nn_hdim)#This is basically populating an array of weights for each input to each node
	W2 = np.random.randn(nn_hdim, 3) # this is populating  an array of weights for each output to each node
	b1 = np.zeros((1, nn_hdim)) #
	b2 = np.zeros((1, 3))
	model = {}
	for i in range(num_passes): #loop until w and b are minimized
		a = X.dot(W1) + b1
		h = np.tanh(a)
		z = h.dot(W2) + b2

		raiseToe = np.exp(z)
		total = np.sum(raiseToe, axis =1, keepdims = True)
		softmax = raiseToe / total #calculate softmax value at node
		
		softmax[range(len(X)), y] -= 1 #decrement layer
		

		PartialW2 = np.transpose(h) #Transpose so that the we multiply each node by its matching weight. Does not work without
		PartialW2 = PartialW2.dot(softmax)
		BiasLayer2 = np.sum(softmax, axis = 0, keepdims = True)
		
		loss = 1 - np.power(h, 2)
		WT2 = np.transpose(W2) #Transpose so that we multiply each node by its matching weight. Won't work without it
		propagatedLoss = softmax.dot(WT2) * loss
		
		WT1 = np.transpose(X)#Transpose so that the we multiply each node by its matching weight. Does not work without
		PartialW1 = np.dot(WT1, propagatedLoss)
		BiasLayer1 = np.sum(propagatedLoss, axis = 0) 


		W1 = W1 - (eta* PartialW1)#decrement weights and biases as you learn until loss is minimized
		W2 = W2 - (eta* PartialW2)
		b1 = b1 - (eta* BiasLayer1)
		b2 = b2 - (eta* BiasLayer2)

		model = {'W1': W1, 'b1': b1, 'W2' : W2, 'b2':b2}

		if print_loss == True and num_passes % 1000 == 0:
			print(calculate_loss(model, X, y))

	return model

###########################################################################################################
# def plot_decision_boundary(pred_func, X, y)
# Plots the data and decision boundary
# inputs: pred_func, X, y
# returns: nothing
############################################################################################################

def plot_decision_boundary(pred_func, X, y):
	x_min, x_max = X[:, 0].min()-.5, X[:, 0].max() + .5
	y_min, y_max = X[:, 1].min() -.5, X[:, 1].max() +.5
	h = 0.01

	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)

	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
	plt.scatter(X[:, 0], X[:, 1], c= y, cmap = plt.cm.Spectral)




	

	








