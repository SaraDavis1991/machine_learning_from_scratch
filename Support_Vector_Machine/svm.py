###################################################################################################################################################
# filename: svm.py
# author: Sara Davis 
# date: 12/9/2018
# version: 1.0
# description: 
##################################################################################################################

import numpy as np 
import matplotlib.pyplot as plt 

###########################################################################################################
# def svm_train_brute(training)
# This function trains a linear SVM
# inputs: training
# returns: w,b, S
############################################################################################################
def svm_train_brute(training):
	positiveList = [] #create a numpy array of 2 col 1 row
	negativeList = []

	count = 0
	for points in range(len(training)):
	
		if training[points][2] == 1:
			positiveList.append(training[points])
		if training[points][2] == -1:
			negativeList.append(training[points])

	positiveSet = np.zeros(shape = (len(positiveList), 2))
	negativeSet = np.zeros(shape = (len(negativeList), 2))

	for rows in range(len(positiveList)):
		for cols in range(2):
			positiveSet[rows][cols] = positiveList[rows][cols]
		
	for rows in range(len(negativeList)):
		for cols in range(2):
			negativeSet[rows][cols] = negativeList[rows][cols]
	
	smallestDistance = 1000000
	where = []
	subset = []
	for rowsP in range(len(positiveSet)):
		for rowsN in range(len(negativeSet)):
			a = positiveSet[rowsP][0] - negativeSet[rowsN][0]
			b = positiveSet[rowsP][1] - negativeSet[rowsN][1]
			cSq = a**2 + b**2
			c = cSq**(.5)
			if c < smallestDistance:
				smallestDistance = c
				
				subset.append(rowsP)
				subset.append(rowsN)
				where = []
				where.append(subset)
				subset = []
				
			if c == smallestDistance:
				subset.append(rowsP)
				subset.append(rowsN)

				for i in range(len(where)):
					if where[i] == subset:
						subset = []
				if len(subset) > 0:
					where.append(subset)
				subset = []

	if len(where) > 1:
		wDir = []
		S = np.zeros((3,2))
		if where[0][0] != where[1][0]:
			SV1 = positiveSet[where[0][0]]#pos
			SV2 = positiveSet[where[1][0]]#pos
			SV3 = negativeSet[where[0][1]]

			midX = (SV3[0] + SV2 [0]) / 2
			midY = (SV3[1] + SV2[1]) /2
			
			secondMidX = (SV3[0] + SV1[0]) /2
			secondMidY = (SV3[1] + SV1[1])/2

			

			w1 = secondMidX - midX
			w2 = secondMidY - midY
		

			w= np.zeros((1,2))
			w[0][0] = w1
			w[0][1] = w2


			wX = midY - (midX * (w1/w2))
			b =  wX
			

		if where[0][1] != where[1][1]:
			SV1 = positiveSet[where[0][0]] #pos
			SV2 = negativeSet[where[0][1]]#neg
			SV3 = negativeSet[where[1][1]]
			

			midX = SV1[0] + SV2 [0] / 2
			midY = SV1[1] + SV2[1] /2
			
			secondMidX = SV3[0] + SV1[0] /2
			secondMidY = SV3[1] + SV1[1]/2

			
			w1 = secondMidX - midX
			w2 = secondMidY - midY
			

			w= np.zeros((1,2))
			w[0][0] = w1
			w[0][1] = w2


			wX = midY - (midX * (w1/w2))
			b =  wX
		S[0] = SV1
		S[1] = SV2
		S[2] = SV3
	else:
		wDir = []
		S = np.zeros((2,2))
		SV1 = positiveSet[where[0][0]]
		SV2 = negativeSet[where[0][1]]
		

		yChange = SV2[1] - SV1[1]
		xChange = SV2[0] - SV1[0]
		midY = (SV1[1] + SV2[1])/2
		midX = (SV1[0] + SV2[0])/2
	

		w = np.zeros((1,2))
		if xChange == 0:
			val = yChange/2
			
			w[0][0] = 0
			w[0][1] = val
		
		elif yChange ==0:
			val = xChange/2
			
			w[0][0] = val
			w[0][1] = 0

		else:
			w[0][0] = -yChange
			w[0][1] = xChange

		b = midY - (midX * yChange )
	

		S[0] = SV1
		S[1] = SV2

	
	badPts = []
	for i in range(len(training)):
	
		check = training[i][0] *w[0][0] + b
		if check >= training[i][1]* training[i][2]:
			badPts.append(training[i])
		
	if len(badPts) ==2:
		S = np.zeros((2,2))
		SV1[0][0] = badPts[0][0]
		SV1[0][1] = badPts[0][1]
		SV2[0][0] = badPts[1][0]
		SV2[0][1] = badPts[1][1]

		yChange = SV1[0][1] - SV2[0][1]
		xChange = SV1[0][0] - SV2[0][0]

		midX = (SV1[0][0] + SV2[0][0] )/2
		midY = (SV1[0][1] + SV2[0][1]) /2

		w = np.zeros((1,2))
		w[0][0] = yChange
		w[0][1] = xChange

		b = midY - (midX * yChange )

		S[0][0] = SV1[0][0]
		S[0][1] = SV1[0][1]
		S[1][0] = SV2[0][0]
		S[1][1] = SV2[0][1]



	if len(badPts) ==3:
		S = np.zeros((3,2))
		SV1[0][0] = badPts[0][0]
		SV1[0][1] = badPts[0][1]
		SV2[0][0] = badPts[1][0]
		SV2[0][1] = badPts[1][1]
		SV3[0][0] = badPts[2][0]
		SV3[0][1] = badPts[2][1]
		midX = SV3[0] + SV2 [0] / 2
		midY = SV3[1] + SV2[1] /2
		
		secondMidX = SV3[0] + SV1[0] /2
		secondMidY = SV3[1] + SV1[1]/2

		w1 = secondMidX - midX
		w2 = secondMidY - midY
		w= np.zeros((1,2))
		w[0][0] = w1
		w[0][1] = w2

		wX = w1 * SV1[0] + w2 * SV1[1]
		b = 1 - wX

		S = badPts
	
	#print(smallestDistance)
	return(w,b, S)

###########################################################################################################
# def distance_point_to_hyperplane(pt, w, b)
# Calculate the distance of a point to the hyperplane
# inputs: pt, w, b
# returns:distance
############################################################################################################
def distance_point_to_hyperplane(pt, w, b):
	if w[0][1] != 0:

		distance = pt[1] - (w[0][0] * pt[0] + b) #does not work when w is 1 (always on line) 

	else:
		distance = pt[0] - (w[0][1]* pt[0] + b)

	return distance
###########################################################################################################
# def compute_margin(training, w, b)
# compute the margin between the support vectors and the boundary
# inputs: training, w, b
# returns: smallestDistance
############################################################################################################
def compute_margin(training, w, b):
	positiveList = [] #create a numpy array of 2 col 1 row
	negativeList = []

	count = 0
	for points in range(len(training)):
	
		if training[points][2] == 1:
			positiveList.append(training[points])
		if training[points][2] == -1:
			negativeList.append(training[points])

	positiveSet = np.zeros(shape = (len(positiveList), 2))
	negativeSet = np.zeros(shape = (len(negativeList), 2))

	for rows in range(len(positiveList)):
		for cols in range(2):
			positiveSet[rows][cols] = positiveList[rows][cols]
		
	for rows in range(len(negativeList)):
		for cols in range(2):
			negativeSet[rows][cols] = negativeList[rows][cols]
	
	smallestDistance = 1000000
	where = []
	subset = []
	for rowsP in range(len(positiveSet)):
		for rowsN in range(len(negativeSet)):
			a = positiveSet[rowsP][0] - negativeSet[rowsN][0]
			b = positiveSet[rowsP][1] - negativeSet[rowsN][1]
			cSq = a**2 + b**2
			c = cSq**(.5)
			if c < smallestDistance:
				smallestDistance = c


	return smallestDistance

###########################################################################################################
# def svm_test_brute(w, b, x)
# return 1 if correct, -1 if incorrect
# inputs: w, b , x
# returns: 1, -1
############################################################################################################
def svm_test_brute(w, b, x):

	for i in range(len(X)):
		if x[0] * w[i][0] < x[1]:
			return 1
		else:
			return -1
