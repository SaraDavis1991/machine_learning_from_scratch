###################################################################################################################################################
# filename: clustering.py
# author: Sara Davis 
# date: 9/25/2018
# version: 2.0
# description: Performs K-Means Clustering
###################################################################################################################################################

import numpy as np
import math
import random

###########################################################################################################
# def K_Means(X, K)
# Randomly initialize the cluster centers, then assign data points to each center
# inputs: X, K
# returns: centers (cluster centers)
############################################################################################################
def K_Means(X, K):
	count = 0
	num_dimensions = len(X[0])

	cluster = [0] * len(X)
	
	
	prev_cluster = [-1] * len(X)

	
	centers = []
	for i in range(K):
		next_cluster = []
		
		
		centers +=[random.choice(X)]
		
		find_another = False
	
	while (cluster != prev_cluster)  and (find_another == False):
		prev_cluster = list(cluster)
		
		find_another = False
		
		
		for point in range (len(X)):

			shortest = float (100000000.0) #randomly chosen float distance. Large enough that clustering would not be efficient
			
			
			for centerPoint in range (len(centers)):
				distance = linearDistance(X[point], centers[centerPoint], X)
					
				for x in range (len(distance)):
					if (distance[x] < shortest):
						shortest =distance[x]
						cluster[point] = centerPoint #element distance belongs to specific cluster
						
		
		for j in range (len(centers)):
			next_center = [0] * num_dimensions
			items = 0
			for point in range(len(X)):
				if (cluster[point] == j):
					for m in range(num_dimensions):
						next_center[m] += X[point][m]
						items += 1
			for n in range(num_dimensions):
				if items != 0:
					next_center[n] = next_center[n] / float (items)
					
				else:
					new_center = random.choice(X)
					find_another = True
			centers[j] = next_center
			
	print("Cluster centers" + str(centers))
	return(centers)	

###########################################################################################################
# def linearDistance(point1, point2, X)
# Calculate the distance of two points to the center
# inputs: point1, point2, X
# returns: distList (list of distances of points to centers)
############################################################################################################
def linearDistance(point1, point2, X):
	distList = []
	length = 0
	dist = 0
	

	#calculate linear distance between test point and centers
	
 
	for m in range (len(X[0])):#allows n dimensional
		length = np.square(point1[m] - point2[m])
		
		dist += length
		
	dist = np.sqrt(dist)
	distList.append(dist)#add to end of list
	
		
	return distList

###########################################################################################################
# def K_Means_better(X, K)
# Find the best center
# inputs: X, K
# returns: centers[0]
############################################################################################################
def K_Means_better(X, K):

	Centers = []
	nextCenter = []
	Centers.append(K_Means(X, K))
	nextCenter.append(K_Means(X,K))



	if Centers != nextCenter:
		K_Means_better(X, K)
	else:	
		print("Optimized center" + str(Centers[0]))
		return Centers[0]
