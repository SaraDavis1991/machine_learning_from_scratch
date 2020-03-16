###################################################################################################################################################
# filename: compress.py
# author: Sara Davis 
# date: 12/5/2018
# version: 1.0
# description: Compress images
###################################################################################################################################################

import pca
import os
import matplotlib.pyplot as plt
import numpy as np

###########################################################################################################
# def compress_images(DATA, k)
# This function takes images, calls pca function, and compresses the image
# inputs: model, X, y
# returns: totalLoss
############################################################################################################
def compress_images(DATA, k):

	Z = pca.compute_Z(DATA, centering = True, scaling = False)
	COV = pca.compute_covariance_matrix(Z)
	L, PCS = pca.find_pcs(COV)
	Z_Star = pca.project_data(Z, PCS, L, k, 0)
	PCS = PCS[:, :k]


	XCompressed = Z_Star.dot(np.transpose(PCS))



	path = 'Output'
	if os.path.exists(path) == False:
	
		os.mkdir(path)



	for col in range(len(XCompressed[0])):
		XCompressed[:, col] = ((XCompressed[:, col] - XCompressed[:, col].min()) * 1/(XCompressed[:, col].max()-XCompressed[:,col].min()) * 255).astype(float)#should this be rescaling cols separately? If so , HOW

	tempArray = np.zeros((len(XCompressed), 1)) 
	colData = 0
	rowData = 0
	

	for items in range(len(XCompressed[1])):
		for row in range(len(tempArray)):
				
			tempArray[row][0] = XCompressed[row][items] #possibly row??
				
		here = str(path) + "/" + str(items) + ".jpg"
		

		imArray = tempArray.reshape((r,c))

		plt.imsave(here, imArray, format = "jpg", cmap = 'gray')

###########################################################################################################
# def load_data(input_dir)
# This function loads the data
# inputs: input_dir
# returns: DATA
############################################################################################################
def load_data(input_dir):
	imageList = os.listdir(input_dir)
	imageList.sort()
	check = plt.imread(str(input_dir)+ "/" + str(imageList[0]))
	global r 
	global c
	r = len(check)
	c = len(check[0])

	total = r * c
	DATA = np.zeros(shape = (total, len(imageList)))

	colData = 0
	for image in imageList:
		
		rowData = 0
		imArray = plt.imread(str(input_dir)+ "/" + str(image))

		for row in range(len(imArray)):
			for col in range (len(imArray[0])):
				DATA[rowData][colData] = imArray[row][col]
				rowData +=1
		colData+=1
				
	DATA.astype(float)


	return DATA
