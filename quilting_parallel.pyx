#cython: boundscheck=False, wraparound=False

cimport numpy as np
import numpy as np
cimport cython
from cython.parallel import parallel, prange
from libc.math cimport sqrt
from libc.stdlib cimport rand
from libc.stdint cimport uintptr_t

import math
import sys
import os
import itertools
import random

from PIL import Image

# numpy types
# TODO need all?
ctypedef np.float32_t FLOAT
ctypedef np.uint32_t UINT
ctypedef np.int32_t INT


cdef pastePatches(int textureSize, int tileSize, int overlap, int numRows, int numCols, int tid,
				  INT[:,:,:,:] patches, INT[:,:,:] initialPatch):
	# TODO: necessary?
	cdef:
		int i, j, rowNo, colNo, chosenIdx
		# int refPatchLeftX = refPatchLeft.shape[1]
		# int refPatchLeftY = refPatchLeft.shape[0]
		# int refPatchUpX = refPatchUp.shape[1]
		# int refPatchUpY = refPatchUp.shape[0]
		# int refPatchBothX = refPatchBoth.shape[1]
		# int refPatchBothy = refPatchBoth.shape[0]
		bool blockLeft, blockUp

	rowNo = tid / numCols
	colNo  = tid % numCols
	print "On iteration %i" % tid

	# declaring distance arrays
	distLeft = np.zeros(patches.shape[0], dtype=np.float32)
	distUp = np.zeros(patches.shape[0], dtype=np.float32)
	distBoth = np.zeros(patches.shape[0], dtype=np.float32)
	# TODO: double?
	distances = np.empty_like(patches, dtype=np.float32)

	cdef:
		FLOAT[:] d = distLeft
		FLOAT[:,:,:] refPatchLeft, refPatchUp, refPatchBoth
		INT[:,:,:] chosenPatch
		FLOAT[:,:] costMap, pathMaskLeft, pathMaskUp, pathMaskBoth

	with nogil:
		blockLeft = colNo>0
		blockUp = rowNo>0
		
		# find reference patchs and calculate overlap distances over all sample patches
		if blockLeft:
			refPatchLeft = texture[rowNo*tileSize:min(rowNo*tileSize + patchSize, textureSize[1]), 
							colNo*tileSize:min(colNo*tileSize + overlap, textureSize[0]), :]
			overlapDistances(refPatchLeft, patches, distances, distLeft)
			d = distLeft

		if blockUp:
			refPatchUp = texture[rowNo*tileSize:min(rowNo*tileSize + overlap, textureSize[1]), 
							colNo*tileSize:min(colNo*tileSize + patchSize, textureSize[0]), :]
			overlapDistances(refPatchUp, patches, distances, distUp)
			d = distUp

		if blockLeft and blockUp:
			refPatchBoth = texture[rowNo*tileSize:min(rowNo*tileSize + overlap, textureSize[1]), 
							colNo*tileS           ize:min(colNo*tileSize + overlap, textureSize[0]), :]
			overlapDistances(refPatchBoth, patches, distances, distBoth)
			d = distLeft + distUp - distBoth

		# finds appropriate random patch
		chosenIdx = getMatchingPatch(d, 1.1)
		chosenPatch = patches[chosenIdx, :, :, :]

		# determines minimum cut boundary and overlays onto chosen patch
		if blockLeft:
			costMap = makeCostMap(refPatchLeft, chosenPatch[:refPatchLeft.shape[0], :overlap, :])
			pathMaskLeft = cheapVertCut(costMap)
			# TODO write our own function for this
			overlapLeft = combineRefAndChosen(pathMaskLeft, refPatchLeft, chosenPatch, 0, overlap)
			# overlapLeft = np.where(np.dstack([pathMaskLeft] * 3), refPatchLeft, chosenPatch[:refPatchLeft.shape[0], :overlap, :])
			# overwrite with min cut
			chosenPatch[:refPatchLeft.shape[0],:overlap,:] = overlapLeft

		if blockUp:
			# chosenSize = min(colNo*tileSize + patchSize, textureSize[0]) - colNo*tileSize
			# TODO: stupid solution; find better one
			costMap = makeCostMap(refPatchUp, chosenPatch[:overlap, :refPatchUp.shape[1], :])
			pathMaskUp = cheapHorizCut(costMap)
			overlapUp = combineRefAndChosen(pathMaskUp, refPatchUp, chosenPatch, 1, overlap)
			# overlapUp = np.where(np.dstack([pathMaskUp] * 3), refPatchUp, chosenPatch[:overlap, :refPatchUp.shape[1], :])
			# overwrite with min cut
			chosenPatch[:overlap,:refPatchUp.shape[1],:] = overlapUp

		if blockLeft and blockUp:
			# pathMaskBoth = np.zeros((refPatchUp.shape[0], refPatchLeft.shape[1]))
			for i in range(refPatchUp.shape[0]):
				for j in range(refPatchLeft.shape[1]):
					# bitwise or operation
					pathMaskBoth[i][j] = 1 - ((1-pathMaskUp[i][j]) * (1-pathMaskLeft[i][j]))

			pathMaskLeft[:pathMaskBoth.shape[0],:] = pathMaskBoth
			pathMaskUp[:,:pathMaskBoth.shape[1]] = pathMaskBoth

			overlapBothLeft = combineRefAndChosen(pathMaskLeft, refPatchLeft, chosenPatch, 0, overlap)
			overlapBothUp = combineRefAndChosen(pathMaskUp, refPatchUp, chosenPatch, 1, overlap)
			# overlapBothLeft = np.where(np.dstack([pathMaskLeft] * 3), refPatchLeft, chosenPatch[:refPatchLeft.shape[0], :overlap, :])
			# overlapBothUp = np.where(np.dstack([pathMaskUp] * 3), refPatchUp, chosenPatch[:overlap, :refPatchUp.shape[1], :])
			
			# overwrite with min cut
			chosenPatch[:refPatchLeft.shape[0],:overlap,:] = overlapBothLeft
			chosenPatch[:overlap,:refPatchUp.shape[1],:] = overlapBothUp

		insert(texture, chosenPatch, rowNo*tileSize, colNo*tileSize)


'''
This function computes the distance of refPatch to all patches in patches, returning a 1D array of all distances.
Returns 1-D array of distances over all patches.
'''
# TODO: should make contiguous?
cdef overlapDistances(FLOAT[:,:,:] refPatch,
					   INT[:,:,:,:] patches,
					   FLOAT[:,:,:,:] distances,
					   FLOAT[:] results) nogil:
	cdef:
		int numPatches = patches.shape[0]
		int i, j, k, p
	
	# calculate distances of refPatch from patches
	# TODO: try load balancing with chunksize
	for i in prange(numPatches, num_threads=8, schedule='dynamic'):
		for j in range(refPatch.shape[0]):
			for k in range(refPatch.shape[1]):
				for p in range(3): # num channels
					distances[i,j,k,p] = patches[i,j,k,p] - refPatch[j,k,p]

	#calculate L2 norm and sum over all reference patch pixels
	for i in prange(numPatches, num_threads=8, schedule='dynamic'):
		for j in range(refPatch.shape[0]):
			for k in range(refPatch.shape[1]):
				results[i] += sqrt(distances[i,j,k,0]**2 + distances[i,j,k,1]**2 + distances[i,j,k,2]**2)


cdef combineRefAndChosen(FLOAT[:,:] pathMask, 
						FLOAT[:,:,:] refPatch, 
						FLOAT[:,:,:] chosenPatch, 
						int dir, 
						int overlap) nogil:
	cdef:
		# TODO: do we want this function to return or be void???
		FLOAT[:,:,:] overlap = refPatch

	# dir: 0 for left, 1 for up
	# overlapLeft = np.where(np.dstack([pathMaskLeft] * 3), refPatchLeft, chosenPatch[:refPatchLeft.shape[0], :overlap, :])
	if dir == 0:
		for i in range(refPatch.shape[0]):
			for j in range(overlap):
				# use refPatch if 1; chosenPatch if 0
				if pathMask[i][j] == 1:
					overlap[i][j] = refPatch[i][j]
				else:
					overlap[i][j] = chosenPatch[i][j]
	else:
		for i in range(overlap):
			for j in range(refPatch.shape[1]):
				# use refPatch if 1; chosenPatch if 0
				if pathMask[i][j] == 1:
					overlap[i][j] = refPatch[i][j]
				else:
					overlap[i][j] = chosenPatch[i][j]

	return overlap

# TODO: cythonize all helper functions
def makePatches(img, patchSize):
	'''
	This function takes in an img with size img.shape and a patch size patchSize, returns an array of shape
	(patchSize, patchSize, #num of patches), so (:,:,idx) returns the idx'th patch
	'''

	# check that img should have channel axis, so (x,y,channel)
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


cdef getMatchingPatch(FLOAT[:] distances,
					  FLOAT thresholdFactor) nogil:
	'''
	Given a 1-D array of patch distances, choose matching patch index that is within threshold.
	'''
	cdef:
		FLOAT[:] d = distances 
		int i
		FLOAT minVal

	# do not select current patch
	minVal = 99999.
	for i in range(len(d)):
		if d[i] < minVal and d[i] > 0.01:
			minVal = d[i]

	cdef:
		FLOAT threshold = thresholdFactor * minVal
		int ctr = 0

	# choose random index such that the distance is within threshold factor of minimum distance
	# TODO: make default thresholdFactor
	threshold = thresholdFactor * m
	
	# count number of qualifying indices to allocate memory for indices
	for i in range(len(d)):
		if d[i] < threshold:
			ctr += 1

	cdef:
		FLOAT[ctr] indices

	# store all qualifying indices of d in indices
	for i in range(len(d)):
		if d[i] < threshold:
			indices[ctr - 1] = i
			ctr -= 1
	
	cdef:
		int r = rand()
		int idx = r/ctr

	# indices = np.where(d < threshold)[0]
	# idx = indices[np.random.randint(0,len(indices))]
	return indices[idx]


cpdef insert(target, patch, i, j):
	'''
	This function inserts a patch into img at position (i,j).
	'''
	patchSize = patch.shape[0]
	patchV = min(i+patchSize, target.shape[0]) - i
	patchH = min(j+patchSize, target.shape[1]) - j
	target[i:min(i+patchSize, target.shape[0]), j:min(j+patchSize, target.shape[1]), :] = patch[:patchV, :patchH, :]


cdef makeCostMap(FLOAT[:,:,:] img1,
				 FLOAT[:,:,:] img2) nogil:
	'''
	This function takes in 2 overlapping image regions, computes pixel-wise L2 norm and returns cost map.
	'''
	cdef:
		int i,j,k
		FLOAT[:,:,:] distances = img1
		FLOAT[img1.shape[0],img1.shape[1]] results = 0.

	# calculate distances of refPatch from patches
	for i in range(img1.shape[0]):
		for j in range(img1.shape[1]):
			for k in range(3): # num channels
				distances[i,j,k] = img[i,j,k] - img[i,j,k]

	#calculate L2 norm and sum over all reference patch pixels
	for i in range(img1.shape[0]):
		for j in range(img1.shape[1]):
			results[i][j] += sqrt(distances[i,j,0]**2 + distances[i,j,1]**2 + distances[i,j,2]**2)
	
	return results
	# return np.sqrt(np.sum(np.square(img1-img2), axis=2))


'''
DP this shit, yo
'''
cdef calcMinCosts(FLOAT[:,:] costMap) nogil:	
	cdef:
		# TODO does this copy or reference?
		# this needs to be a copy
		#FLOAT [costMap.shape[0]][costMap.shape[1]] cumuCosts
		FLOAT[:,:] cumuCosts = costMap
		int x = costMap.shape[1]
		int y = costMap.shape[0]
		FLOAT[3] c
		int i, j
	
	# default ones
	cumuCosts[:,:] = 1.
	#cumuCosts = np.ones(costMap.shape)
	
	#cumuCosts[:] = costMap[:]

	for i in range(y - 1):
		for j in range(x):
			if j == 0:
				c = cumuCosts[i, 0:2]
			elif j == x - 1:
				c = cumuCosts[i, x - 2:x]
			else:
				c = cumuCosts[i, j - 1:j + 2]

			# TODO find min of c without np
			cumuCosts[i + 1,j] += np.min(c)

	return cumuCosts


cdef pathBacktrace(cumuCosts):
	'''
	trace DP shit backwards, yo
	'''
	x = cumuCosts.shape[1]
	y = cumuCosts.shape[0]

	pathCosts = np.zeros(cumuCosts.shape, dtype='float32')

	minIdx = 0
	maxIdx = x - 1
	for row in range(y - 1, -1, -1):
		i = np.argmin(cumuCosts[row, minIdx:maxIdx + 1])
		pathCosts[row, i] = 1
		minIdx = np.max([0, i - 1])
		maxIdx = np.min([x - 1, i + 1])

	return pathCosts


cdef cheapVertPath(FLOAT[:,:] costMap) nogil:
	cdef:
		FLOAT[:,:] costs = calcMinCosts(costMap)
		FLOAT[:,:] path = pathBacktrace(costs)
	return path


'''
Generate binary mask
'''
cdef cheapVertCut(FLOAT[:,:] costMap) nogil:
	cdef:
		FLOAT[:,:] path = cheapVertPath(costMap)

	for row in range(path.shape[0]):
		path[row, 0:np.argmax(path[row, :])] = 1
	return path

# TODO oh god taking a transpose in C...
cdef cheapHorizCut(costMap):
	path = cheapVertCut(costMap.T).T
	return path



























