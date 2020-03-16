###################################################################################################################################################
# filename: testGradientDescent.py
# author: Sara Davis 
# date: 10/12/2018
# version: 1.0
# description: runs GradientDescent
#############################################################################################################################################
import numpy as np
import math
from gradient_descent import gradient_descent

def df(x):
	#return 2*x

	return np.array([2*x[0] -6 , 2*x[1] - 4])
	#return np.array([2*x[0], 2*x[1], 2*x[2]])
	#return np.array([4*math.pow(x[0], 3), 6*x[1]])

#x = gradient_descent(df, np.array([5.0]), 0.1)
x = gradient_descent(df, np.array([5.0, 5.0]), 0.1)
#x = gradient_descent(df, np.array([5.0, 5.0, 5.0]), 0.1)
#x = gradient_descent(df, np.array([1.0, 1.0]), 0.01)

print(x)
