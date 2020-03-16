###################################################################################################################################################
# filename: gradient_descent.py
# author: Sara Davis 
# date: 10/11/2018
# version: 1.0
# description: Performs gradient descent
###################################################################################################################################################

def gradient_descent(df, x, n):
	steps = 0
	length = x.size
	gradient = []
		
	for i in range(length):
		gradient.append(1)
		
	for i in range(length):
		
		while True:
			gradient = df(x)
			

			if (gradient[i] <=0.0001):
				break
				
			x[i] = x[i] - (n*gradient[i])
			steps += 1
	
	return x
