###################################################################################################################################################
# filename: perceptron.py
# author: Sara Davis 
# date: 10/8/2018
# version: 2.0
# description: Performs the perceptron algorithm
###################################################################################################################################################

import numpy as np
###########################################################################################################
# def perceptron_train(X,Y)
# This function trains the perceptron
# inputs: X,Y
# returns: perceptron_check(X, Y, weight_array, bias, total, counter, i) - this is the weights and bias
############################################################################################################
def perceptron_train(X,Y):
	bias = 0
	weight_array = []
	total = 0
	counter = 0
	i = 0
	for j in range(len(X[0])):
		weight_array.append(0)

	return perceptron_check(X, Y, weight_array, bias, total, counter, i)
###########################################################################################################
# def perceptron_check(X,Y, weight_array, bias, total, counter, i)
# This function updates the weights and bias if the perceptron misclassifies samples
# inputs: X, Y, weight_array, bias, total, counter, i
# returns: a list of weight_array and bias
############################################################################################################
def perceptron_check(X, Y, weight_array, bias, total, counter, i):
	
	while counter in range(len(X)): 
		while i < len(X):			
			for j in range(len(X[0])):
				total+=(X[i][j] * weight_array[j])

			total+=bias

			if Y[i][0] == 0:
				Y[i][0] = -1

			check = total * Y[i][0]
			total = 0
			
			if check <= 0:
				weight_array, bias = update_perceptron(X, Y, weight_array, bias, i)
				counter = 0
				i+= 1	
			else:
				counter +=1	
				i+= 1
			if i == len(X) and counter in range(len(X)):
				i = 0	
	return[weight_array, bias]
###########################################################################################################
# def update_perceptron(X, Y, weight_array, bias, i)
# This function updates the weight array and bias of the perceptron
# inputs: X, Y, weight_array, bias, total, counter, i
# returns: a list of weight_array and bias
############################################################################################################
def update_perceptron(X, Y, weight_array, bias, i):
	bias += Y[i][0]
	
	for j in range(len(X[0])):
		
		weight_array[j] += X[i][j] * Y[i][0]
	return weight_array, bias
###########################################################################################################
# def perceptron_test(X_test, Y_test, w, b)
# This function calls predict_label, compare_label, and accuracy
# inputs: X_test, Y_test, w, b
# returns: accuracy
############################################################################################################
def perceptron_test(X_test, Y_test, w, b):
	length = len(Y_test)
	prediction_array = predict_label(X_test, w, b)
	counter = compare_label(prediction_array, Y_test)
	return accuracy(counter, length)
###########################################################################################################
# def predict_label(X_test, w, b)
# Use the perceptron the predict the label of a test sample
# inputs: X_test, Y_test, w, b
# returns: predictions for all samples
############################################################################################################
def predict_label(X_test, w, b):
	prediction_array = []
	total = 0
	
	for k in range(len(X_test)):
		prediction_array.append(0)
	
	for i in range (len(X_test)):
		for j in range(len(X_test[0])):
			total +=(X_test[i][j] *w[j])
		
		total +=b
		
		
		if total <= 0:
			prediction_array[i] =-1
		if total > 0:
			prediction_array[i] =  1
		total = 0
	return prediction_array
###########################################################################################################
# def compare_label(prediction_array, Y_test)
# Compare the predictions to the test sample groundtruths
# inputs: prediction_array, Y_test
# returns: counter (number incorrectly classified)
############################################################################################################
def compare_label(prediction_array, Y_test):
	length = len(prediction_array)
	counter = 0
	for i in range(len(prediction_array)):
		if prediction_array[i] == Y_test[i][0]:
			counter += 1
	
	return counter
###########################################################################################################
# def accuracy(counter, length)
# Calculate accuracy
# inputs: counter, length
# returns: percent
############################################################################################################	
def accuracy(counter, length):
	percent = (counter /length) * 100
	return percent
		
