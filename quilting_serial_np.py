#!/usr/bin/python
'''
Helper functions for serial impementation of the Efros-Freeman image quilting algorithm,
using Numpy

Authors:
Aidi Zhang
Samuel Cheng

Harvard CS205, fall 2015
Course Head: Prof. Thouis (Ray) Jones
Teaching Fellow: Kevin Chen

'''

import numpy as np
import math
import random


'''
This function computes the distance of refPatch to all patches in patches
Returns 1-D array of distances over all patches.
'''
def overlapDistances(refPatch, patches):
	# find refPatch area of each sample patch
	ov = patches[:, :refPatch.shape[0], :refPatch.shape[1], :]

	# find distances of refPatch area of sample patches from refPatch itself
	numPatches = patches.shape[0]
	distances = ov - np.tile(refPatch, (numPatches, 1, 1, 1))

	# calculate L2 norm and sum over all reference patch pixels
	# TODO #1: parallelize
	distances = np.sqrt(np.sum(np.square(distances), axis=3))
	distances = np.sum(distances, axis=1)
	distances = np.sum(distances, axis=1)
	
	return distances


'''
This function takes in an img with size img.shape and a patch size patchSize,
returns an array of shape (patchSize, patchSize, #num of patches),
so (:,:,idx) returns the idx'th patch
'''
def makePatches(img, patchSize):
	#check that img should have channel axis, so (x,y,channel)
	assert img.ndim == 3, "image should have channel axis"

	nX = img.shape[0] - patchSize
	nY = img.shape[1] - patchSize
	nChannels = img.shape[2]
	patches = np.zeros((nX * nY, patchSize, patchSize, nChannels), img.dtype)

	#iterate through all patches from img and store in patches
	k = 0
	for i in range(nX):
		for j in range(nY):
			x,X = i, i + patchSize
			y,Y = j, j + patchSize
			patches[k, :, :, :] = img[x:X, y:Y, :]
			k += 1

	return patches


'''
Given a 1-D array of patch distances, choose matching patch index that is within threshold.
'''
def getMatchingPatch(distances, thresholdFactor):
	d = distances
	# do not select current patch
	d[d < np.finfo(d.dtype).eps] = 99999
	m = np.min(d)
	# choose random index such that the distance is within threshold factor of minimum distance
	# TODO: make default thresholdFactor
	threshold = thresholdFactor * m
	indices = np.where(d < threshold)[0]
	idx = indices[np.random.randint(0, len(indices))]
	return idx


'''
This function inserts a patch into img at position (i,j).
'''
def insert(target, patch, i, j):
	patchSize = patch.shape[0]
	patchV = min(i + patchSize, target.shape[0]) - i
	patchH = min(j + patchSize, target.shape[1]) - j
	target[i:min(i + patchSize, target.shape[0]), \
		   j:min(j + patchSize, target.shape[1]), :] = patch[:patchV, :patchH, :]


'''
This function takes in 2 overlapping image regions, computes pixel-wise L2 norm and returns cost map.
'''
def makeCostMap(img1, img2):
	return np.sqrt(np.sum(np.square(img1 - img2), axis=2))


'''
DP
'''
def calcMinCosts(costMap):
	cumuCosts = np.ones(costMap.shape)
	x = costMap.shape[1]
	y = costMap.shape[0]
	
	cumuCosts[:] = costMap[:]
	for i in range(y - 1):
		for j in range(x):
			if j == 0:
				c = cumuCosts[i, 0:2]
			elif j == x - 1:
				c = cumuCosts[i, x - 2:x]
			else:
				c = cumuCosts[i, j - 1:j + 2]

			cumuCosts[i + 1, j] += np.min(c)

	return cumuCosts

'''
backtrace on DP matrix to find path
'''
def pathBacktrace(cumuCosts):
	x = cumuCosts.shape[1]
	y = cumuCosts.shape[0]

	pathCosts = np.zeros(cumuCosts.shape)

	minIdx = 0
	maxIdx = x - 1
	for row in range(y - 1, -1, -1):
		i = np.argmin(cumuCosts[row, minIdx:maxIdx + 1])
		pathCosts[row, i] = 1
		minIdx = np.max([0, i - 1])
		maxIdx = np.min([x - 1, i + 1])

	return pathCosts


def cheapVertPath(costMap):
	costs = calcMinCosts(costMap)
	path = pathBacktrace(costs)
	return path


'''
Generate binary mask
'''
def cheapVertCut(costMap):
	path = cheapVertPath(costMap)

	for row in range(path.shape[0]):
		path[row, 0:np.argmax(path[row, :])] = 1
	return path


def cheapHorizCut(costMap):
	path = cheapVertCut(costMap.T).T
	return path

