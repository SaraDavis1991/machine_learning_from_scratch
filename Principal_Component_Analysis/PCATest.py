###################################################################################################################################################
# filename: PCATest.py
# author: Sara Davis 
# date: 12/3/2018
# version: 1.0
# description: Run PCA.py
###########################################################################################################################

import pca
import numpy as np 
import compress

X = np.array([[1, 1], [1,-1], [-1, 1], [-1, -1]])
centering = True
scaling = False
Z = pca.compute_Z(X, centering, scaling)
COV = pca.compute_covariance_matrix(Z)
L, PCS = pca.find_pcs(COV)
Z_star = pca.project_data(Z, PCS, L, 1, 0)

X = compress.load_data('/home/sara/Desktop/Data/Train/')
compress.compress_images(X, 100)
