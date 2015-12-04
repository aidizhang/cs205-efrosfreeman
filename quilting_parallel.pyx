#cython: boundscheck=False, wraparound=False

cimport numpy as np
cimport cython
from cython.parallel import parallel, prange
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t

import math
import sys
import os
import png
import itertools
import random

from pylab import *
from PIL import Image
from numpngw import write_png

# numpy types
ctypedef np.float32_t FLOAT

cpdef overlapDistances(FLOAT[:,:,:] refPatch, FLOAT[:,:,:,;] patches):
	'''
	This function computes the distance of refPatch to all patches in patches, returning a 1D array of all distances.
	Returns 1-D array of distances over all patches.
	'''
	cdef:
		FLOAT[:,:,:,:] ov
		FLOAT[:,:,:,:] distances
		FLOAT[] results
		int numPatches

	# find refPatch area of each sample patch
	ov = patches[:,:refPatch.shape[0],:refPatch.shape[1],:]

	# find distances of refPatch area of sample patches from refPatch itself
	numPatches = patches.shape[0]

	# calculate distances of refPatch from patches
	for i in range(numPatches):
		distances[i,:,:,:] = ov[i,:,:,:] - refPatch

	# calculate L2 norm and sum over all reference patch pixels
	# TODO: parallelize distances = np.sqrt(np.sum(np.square(distances), axis=3))
	for i in range(numPatches):
		results[i] = 0
		for j in range(refPatch.shape[0]):
			for k in range(refPatch.shape[1]):
				results[i] += distances[i,j,k,0]**2 + distances[i,j,k,1]**2 + distances[i,j,k,2]**2

	return results


def overlapDistancesOld(refPatch, patches):
	'''
	This function computes the distance of refPatch to all patches in patches, returning a 1D array of all distances.
	Returns 1-D array of distances over all patches.
	'''
	# find refPatch area of each sample patch
	ov = patches[:, :refPatch.shape[0], :refPatch.shape[1], :]

	# find distances of refPatch area of sample patches from refPatch itself
	numPatches = patches.shape[0]

	# paralellize ov - refs
	distances = np.zeros(ov.shape)
	for i in range(numPatches):
		distances[i,:,:,:] = ov[i,:,:,:] - refPatch

	# calculate L2 norm and sum over all reference patch pixels
	# parallelize distances = np.sqrt(np.sum(np.square(distances), axis=3))
	for i in range(numPatches):
		for j in range(refPatch.shape[0]):
			for k in range(refPatch.shape[1]):
				distances[i,j,k,0]**2 + distances[i,j,k,1]**2 + distances # ...

		distances[i,:,:,:] = 

	distances = np.sum(distances, axis=1)
	distances = np.sum(distances, axis=1)
	
	return distances


def makePatches(img, patchSize):
	'''
	This function takes in an img with size img.shape and a patch size patchSize, returns an array of shape
	(patchSize, patchSize, #num of patches), so (:,:,idx) returns the idx'th patch
	'''
	#check that img should have channel axis, so (x,y,channel)
	assert img.ndim == 3, "image should have channel axis"

	nX = img.shape[0] - patchSize
	nY = img.shape[1] - patchSize
	nChannels = img.shape[2]
	patches = np.zeros((nX*nY, patchSize, patchSize, nChannels), img.dtype)

	#iterate through all patches from img and store in patches
	k = 0
	for i in range(nX):
		for j in range(nY):
			x,X = i, i+patchSize
			y,Y = j, j+patchSize
			patches[k,:,:,:] = img[x:X,y:Y,:]
			k += 1

	return patches


def getMatchingPatch(distances, thresholdFactor):
	'''
	Given a 1-D array of patch distances, choose matching patch index that is within threshold.
	'''
	d = distances
	# do not select current patch
	d[d < np.finfo(d.dtype).eps] = 99999
	m = np.min(d)
	# choose random index such that the distance is within threshold factor of minimum distance
	# TODO: make default thresholdFactor
	threshold = thresholdFactor * m
	indices = np.where(d < threshold)[0]
	idx = indices[np.random.randint(0,len(indices))]
	return idx


def mkTexture(textureSize, patches, overlap):
	'''
	Main function
	'''
	patchSize = patches.shape[0]
	nChannels = 3 # currently hardcoded
	tileSize = patchSize - overlap
	texture = np.zeros((textureSize[0], textureSize[1], nChannels), dtype=np.float32)

	nPatches = patches.shape[2]

	k = -1
	width, height = int(math.ceil(textureSize[0]/float(tileSize))), int(math.ceil(textureSize[0]/float(tileSize)))
	for i in range(width):
		for j in range(height):
			k += 1

			#use random patch as first patch
			texture[0:patchSize, 0:patchSize, :] = patches[:,:,:,np.random.randint(0,nPatches)]

			#slicing for left overlap
			sl_l = (slice(i*tileSize, min(i*tileSize + patchSize, texture.shape[0])), 
						slice(j*tileSize, min(j*tileSize + overlap, texture.shape[1])), slice(0, nChannels))

			#slicing for writing PATCH at position (i,j)
			sl_patch = (slice(i*tileSize, min(i*tileSize + patchSize, texture.shape[0])), 
						slice(j*tileSize, min(j*tileSize + patchSize, texture.shape[1])), slice(0, nChannels))

			#finds minimum overlap, and finds minimum distance to available patches
			ov1 = texture[sl_l[0], sl_l[1], :, np.newaxis]
			d = patchDistance(ov1, patches)

            #choose best possible matched patch
			chosenPatchIndex = getMatchingPatch(d)
			chosenPatch =  patches[sl_patch[0], sl_patch[1], :, chosenPatchIndex]

			#paste chosenPatch at texture at position (i,j)
			texture[sl_patch] = chosenPatch

	return texture


def insert(target, patch, i, j):
	'''
	This function inserts a patch into img at position (i,j).
	'''
	patchSize = patch.shape[0]
	patchV = min(i+patchSize, target.shape[0]) - i
	patchH = min(j+patchSize, target.shape[1]) - j
	target[i:min(i+patchSize, target.shape[0]), j:min(j+patchSize, target.shape[1]), :] = patch[:patchV, :patchH, :]


def makeCostMap(img1, img2):
	'''
	This function takes in 2 overlapping image regions, computes pixel-wise L2 norm and returns cost map.
	'''
	return np.sqrt(np.sum(np.square(img1-img2), axis=2))

def calcMinCosts(costMap):
	'''
	DP this shit, yo
	'''
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

			cumuCosts[i + 1,j] += np.min(c)

	return cumuCosts

def pathBacktrace(cumuCosts):
	'''
	trace DP shit backwards, yo
	'''
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

def cheapVertCut(costMap):
	'''
	Generate binary mask
	'''
	path = cheapVertPath(costMap)

	for row in range(path.shape[0]):
		path[row, 0:np.argmax(path[row, :])] = 1
	return path


def cheapHorizCut(costMap):
	path = cheapVertCut(costMap.T).T
	return path



























