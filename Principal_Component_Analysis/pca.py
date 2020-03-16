###################################################################################################################################################
# filename: PCA.py
# author: Sara Davis 
# date: 12/5/2018
# version: 1.0
# description: Perform PCA
###################################################################################################################################################


import numpy as np 
from numpy import linalg as LA
import os
import matplotlib.pyplot as plt

###########################################################################################################
# def compute_Z(X, centering, scaling )
# Computes the the Z matrix for all of the data
# inputs: X, centering, scaling 
# returns: Z
############################################################################################################
def compute_Z(X, centering, scaling ):
	if centering == True:
		average = np.mean(X, axis = 0)
		X_centered = X - average
		X = X_centered
		
		
	if scaling == False:
		std = np.std(X, axis = 0)
		X_std = X/std
		X = X_std
		
		
	Z = X
	
	return Z

###########################################################################################################
# def compute_covariance_matrix(Z)
# Computes the covariance matrix (ZtZ)
# inputs: Z
# returns: COV
############################################################################################################
def compute_covariance_matrix(Z):
	ZT = np.transpose(Z)
	COV = ZT.dot(Z)
	
	return COV
###########################################################################################################
# def find_pcs(COV)
# Computer the principal components
# inputs: COV
# returns: L, PCS
############################################################################################################
def find_pcs(COV):
	def getKey(item): # get specific item for sort
		
		return item[0]

	L, PCS = LA.eig(COV) #compute and split into eigen val and vectors
	
	combine = zip(L, PCS) # combine L and PCS
	combine.sort(key = getKey, reverse = True)#sort list by first item
	nums = zip(*combine) # uncombine L and PCS

	L = np.array(nums[0]) #typecase L and PCS back into array
	PCS = np.array(nums[1])

	return L, PCS
###########################################################################################################
# def project_data(Z, PCS, L, k, var)
# Projects all data into the eigen space using the top k eigen vectors
# inputs: Z, PCS, L, k, var
# returns: Z_Star
############################################################################################################
def project_data(Z, PCS, L, k, var):
	

	counter =1
	if var > 0:
		total = sum (L)
		val = 0	
		
		for i in L:
			val = val + i/total
		
			if val >= var:
				k = counter
				break
			counter+=1
	if k > 0:
		
		Z_Star = Z.dot(PCS)
		Z_Star = Z_Star[:, :k]
	return (Z_Star)


