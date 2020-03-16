###################################################################################################################################################
# filename: nearest_neighbors.py
# author: Sara Davis 
# date: 9/25/2018
# version: 2.0
# description: Performs K-Nearest-Neighbors Clustering
###################################################################################################################################################
import math
import numpy as np
# WRITE A 3, 5, 7 K in, and a calculate best K 
###########################################################################################################
# def KNN_test(X_train, Y_train, X_test, Y_test, K)
# This calls functions to running the clustering algorithm and calculate accuracy
# inputs: X_train, Y_train, X_test, Y_test, K
# returns: percent
############################################################################################################
def KNN_test(X_train, Y_train, X_test, Y_test, K):
	distance = []
	K_Nearest = []
	predict=[]
	access = []
	totalAcc = 0
	accuracy = 0
	counter = 0
	percent =0
	

	
	for j in range(len(X_test)):
		access = X_test[j]
		distance = linearDistance(X_train, X_test, access)
		
		Label = findLabel(distance, K, Y_train)
		counter+=1	
			
		print("\n")
		print("With K =" + str(K) + " neighbors, KNN predicts: " + str(Label)) 
		print("\n")
			
		match = compareLabel(Label, Y_test, counter)
		print("\n")
		print("This test value's label matches the predicted label:" + str(match))
		print("\n")
			
		if match == True:
			accuracy+=1
			
	percent =float((accuracy/len(X_test))*100)
	print("\n")
	print("Accuracy on this set is: " + str(percent))
	print("\n")
	return percent
###########################################################################################################
# def linearDistance(X_train, X_test, access)
# This function calculates linear distance between test point and all training points
# inputs:X_train, X_test, access
# returns: distList (a list of distances from every point to center of cluster)
############################################################################################################		
def linearDistance(X_train, X_test, access):
	distList = []
	length = 0
	dist = 0
	j = 0;

	#calculate linear distance between test point and all training points
	
	for i in range(len(X_train)):
		for k in range(len(X_test[0])): #allows to be n dimensional vector

			length = np.square(access[k] - X_train[i][k])
			dist += length
		dist = np.sqrt(dist)
		distList.append([dist, i])#add to end of list
		distList = sorted(distList) #sort distances by size
	
	
		
	return distList


###########################################################################################################
# def findLabel(distance, K, Y_train)
# Predict the label of a point using the distance of the closest cluster
# inputs:distance, K, Y_train
# returns: Y_train[index]
############################################################################################################			
def findLabel(distance, K, Y_train):
	a = 0
	b = 0
	index = []
	
	for i in range (K):	
		index.append(distance[i][1])

	for i in range(len(index)):
		
			
		item = Y_train[index[0]]
		nextItem = Y_train[index[i]]
			
		if item == nextItem:
			a += 1
		else:
			b += 1
			integer = i	
		
	if a >= b:
		return Y_train[index[0]]
	if b > a:
		return Y_train[index[integer]]
###########################################################################################################
# def compareLabel(Label, Y_test, counter)
# compare the label to the groundtruth
# inputs: Label, Y_test, counter
# returns: True/False
############################################################################################################	
def compareLabel(Label, Y_test, counter):

	#for i in range(len(Y_test)):	
	print("\n")
	print("Y_test= " + str(Y_test[counter-1]))
	print("\n")
	if Label[0]== Y_test[counter-1]:
		return True
	else:
		return False
###########################################################################################################
# def chooseK(X_train, Y_train, X_val, Y_val)
# Choose the value for K
# inputs: X_train, Y_train, X_val, Y_val
# returns: none
############################################################################################################
def chooseK(X_train, Y_train, X_val, Y_val):

	K=1
	accuracy = []

	for i in range (len(X_val)):
		accuracy.append(KNN_test(X_train, Y_train, X_val, Y_val, K))
		K+=1
	K = 1
	

	biggest = accuracy[0]

	
	for j in range(len(X_val)):
		nextNumber = accuracy[j]
		
		if nextNumber>biggest:

			biggest = nextNumber
			K = j+1


	print("The best K value is: " + str(K))		
	print("\n")


	
	
