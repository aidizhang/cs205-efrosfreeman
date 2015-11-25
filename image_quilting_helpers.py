#!/usr/bin/python

import numpy

def verticalPathsCost(costMap):
	'''
	This function takes in a 2D array costMap and outpus for each pixel the cost of the cheapest path from s to t.
	'''	
	pathCosts = numpy.ones(costMap.shape)
	m, n = pathCosts.shape[0], pathCosts.shape[1]
	pathCosts[:] = costMap[:]
	for row in range(m-1):
		for col in range(n):
			if col == 0:	
				c = pathCosts[row,0:2]
			elif col == n - 1:
				c = pathCosts[row,n-2:n]
			else:
				c = pathCosts[row,col-1:col+2]
			pathCosts[row+1, col] += np.min(c)
	return pathCosts

def calculateCost(img1, img2):
	'''
	This function takes in 2 images and calculates pixel-wise L2 norm of the difference image.
	'''
	return np.sqrt(np.sum(np.square(img1-img2), axis=2))