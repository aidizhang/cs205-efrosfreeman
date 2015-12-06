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


cpdef void pastePatch(int textureWidth, int textureHeight, int tileSize, int overlap, int numRows, int numCols, int tid,
				  FLOAT[:,:,:] np_texture, FLOAT[:,:,:,:] patches, INT[:,:,:] initialPatch):
	# TODO: necessary?
	cdef:
		int i, j, chosenIdx
		int numPatches = patches.shape[0]
		int rowNo = tid / numCols
		int colNo = tid % numCols
		int blockLeft, blockUp

	print "On iteration %i" % tid

	# TODO: should entire function be nogil since we're doing some work here?
	# declaring distance arrays
	np_d = np.zeros(patches.shape[0], dtype=np.float32)
	np_distLeft = np.zeros(patches.shape[0], dtype=np.float32)
	np_distUp = np.zeros(patches.shape[0], dtype=np.float32)
	np_distBoth = np.zeros(patches.shape[0], dtype=np.float32)
	# TODO: double?
	np_distances = np.empty_like(patches, dtype=np.float32)
	np_pathCostsLeft = np.zeros((tileSize, overlap), dtype=np.int32)
	np_pathCostsUp = np.zeros((overlap, tileSize), dtype=np.int32)
	np_pathMaskBoth = np.zeros((refPatchUp.shape[0], refPatchLeft.shape[1]), dtype=np.int32)

	np_costMapLeft = np.zeros((tileSize, overlap), dtype=np.float32)
	np_costMapUp = np.zeros((overlap, tileSize), dtype=np.float32)

	cdef:
		FLOAT[:] d = np_d
		FLOAT[:] distLeft = np_distLeft
		FLOAT[:] distUp = np_distUp
		FLOAT[:] distBoth = np_distBoth
		FLOAT[:,:,:,:] distances = np_distances
		INT[:,:] pathCostsLeft = np_pathCostsLeft
		INT[:,:] pathCostsUp = np_pathCostsUp
		INT[:,:] pathMaskBoth = np_pathMaskBoth
		FLOAT[:,:] costMapLeft = np_costMapLeft
		FLOAT[:,:] costMapUp = np_costMapUp
		# why do we have to make a memory view on something we're passing in?
		FLOAT[:,:,:] texture = np_texture
		FLOAT[:,:,:] refPatchLeft, refPatchUp, refPatchBoth
		FLOAT[:,:,:] chosenPatch
	
	with nogil:
		blockLeft = 1 if colNo>0 else 0
		blockUp = 1 if rowNo>0 else 0
		
		# find reference patchs and calculate overlap distances over all sample patches
		if blockLeft:
			refPatchLeft = texture[rowNo*tileSize:int_min(rowNo*tileSize + patchSize, textureHeight), 
							colNo*tileSize:int_min(colNo*tileSize + overlap, textureWidth), :]
			overlapDistances(refPatchLeft, patches, distances, distLeft)
			# reference or actual copy if d = distLeft + distUp - distBoth didn't work
			d = distLeft

		if blockUp:
			refPatchUp = texture[rowNo*tileSize:int_min(rowNo*tileSize + overlap, textureHeight), 
							colNo*tileSize:int_min(colNo*tileSize + patchSize, textureWidth), :]
			overlapDistances(refPatchUp, patches, distances, distUp)
			d = distUp

		if blockLeft and blockUp:
			refPatchBoth = texture[rowNo*tileSize:int_min(rowNo*tileSize + overlap, textureHeight), 
							colNo*tileSize:int_min(colNo*tileSize + overlap, textureWidth), :]
			overlapDistances(refPatchBoth, patches, distances, distBoth)
			# cythonized version of: d = distLeft + distUp - distBoth
			for i in range(numPatches):
				d[i] = distLeft[i] + distUp[i] - distBoth[i]

		# finds appropriate random patch
		chosenIdx = getMatchingPatch(d, 1.1)
		chosenPatch = patches[chosenIdx, :, :, :]

		# determines minimum cut boundary and overlays onto chosen patch
		if blockLeft:
			makeCostMap(refPatchLeft, chosenPatch[:refPatchLeft.shape[0], :overlap, :], costMapLeft)
			cheapVertCut(costMapLeft, pathCostsLeft)
			# TODO: fix?
			combineRefAndChosen(pathCostsLeft, refPatchLeft, chosenPatch, 0, overlap)

		if blockUp:
			# chosenSize = min(colNo*tileSize + patchSize, textureWidth) - colNo*tileSize
			# TODO: stupid solution; find better one
			makeCostMap(refPatchUp, chosenPatch[:overlap, :refPatchUp.shape[1], :], costMapUp)
			cheapHorizCut(costMapUp, pathCostsUp)
			combineRefAndChosen(pathCostsUp, refPatchUp, chosenPatch, 1, overlap)

		if blockLeft and blockUp:
			for i in range(refPatchUp.shape[0]):
				for j in range(refPatchLeft.shape[1]):
					# bitwise or operation
					pathMaskBoth[i,j] = 1 - ((1-pathCostsUp[i,j]) * (1-pathCostsLeft[i,j]))

			pathCostsLeft[:pathMaskBoth.shape[0],:] = pathMaskBoth
			pathCostsUp[:,:pathMaskBoth.shape[1]] = pathMaskBoth

			combineRefAndChosen(pathCostsLeft, refPatchLeft, chosenPatch, 0, overlap)
			combineRefAndChosen(pathCostsUp, refPatchUp, chosenPatch, 1, overlap)

		insert(texture, chosenPatch, rowNo*tileSize, colNo*tileSize)

cdef inline int int_min(int a, int b) nogil: 
	return a if a <= b else b

'''
This function computes the distance of refPatch to all patches in patches, returning a 1D array of all distances.
Returns 1-D array of distances over all patches.
'''
# TODO: should make contiguous?
cdef void overlapDistances(FLOAT[:,:,:] refPatch,
					   FLOAT[:,:,:,:] patches,
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
				# TODO: combine both pranges!
				for p in range(3): # num channels
					distances[i,j,k,p] = patches[i,j,k,p] - refPatch[j,k,p]

	#calculate L2 norm and sum over all reference patch pixels
	for i in prange(numPatches, num_threads=8, schedule='dynamic'):
		for j in range(refPatch.shape[0]):
			for k in range(refPatch.shape[1]):
				results[i] += sqrt(distances[i,j,k,0]**2 + distances[i,j,k,1]**2 + distances[i,j,k,2]**2)


cdef void combineRefAndChosen(INT[:,:] pathMask, 
						FLOAT[:,:,:] refPatch, 
						FLOAT[:,:,:] chosenPatch, 
						int dir,
						int overlap) nogil:
	# dir: 0 for left, 1 for up
	if dir == 0:
		for i in range(refPatch.shape[0]):
			for j in range(overlap):
				# use refPatch if 1; chosenPatch if 0
				if pathMask[i][j] == 1:
					chosenPatch[i][j] = refPatch[i][j]
	else:
		for i in range(overlap):
			for j in range(refPatch.shape[1]):
				# use refPatch if 1; chosenPatch if 0
				if pathMask[i][j] == 1:
					chosenPatch[i][j] = refPatch[i][j]


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
	patches = np.zeros((nX*nY, patchSize, patchSize, nChannels), dtype=np.float32)

	#iterate through all patches from img and store in patches
	k = 0
	for i in range(nX):
		for j in range(nY):
			x,X = i, i+patchSize
			y,Y = j, j+patchSize
			patches[k,:,:,:] = img[x:X,y:Y,:]
			k += 1

	return patches


cdef int getMatchingPatch(FLOAT[:] distances, float thresholdFactor) nogil:
	'''
	Given a 1-D array of patch distances, choose matching patch index that is within threshold.
	'''
	cdef:
		FLOAT[:] d = distances
		int numPatches = distances.shape[0]
		int i
		float minVal

	# do not select current patch
	minVal = 999999.
	for i in range(numPatches):
		if d[i] < minVal and d[i] > 0.01:
			minVal = d[i]

	cdef:
		float threshold = thresholdFactor * minVal
		int ctr = 0

	# choose random index such that the distance is within threshold factor of minimum distance
	# TODO: make default thresholdFactor
	threshold = thresholdFactor * minVal
	
	# count number of qualifying indices to allocate memory for indices
	for i in range(numPatches):
		if d[i] < threshold:
			ctr += 1

	cdef:
		int[ctr] indices

	# store all qualifying indices of d in indices
	for i in range(numPatches):
		if d[i] < threshold:
			indices[ctr - 1] = i
			ctr -= 1
	
	cdef:
		int r = rand()
		int idx = r/ctr

	# indices = np.where(d < threshold)[0]
	# idx = indices[np.random.randint(0,len(indices))]
	return indices[idx]

'''
This function inserts a patch into img at position (i,j).
'''
cpdef void insert(FLOAT[:,:,:] target, FLOAT[:,:,:] patch, int i, int j) nogil:
	cdef:
		int patchSize = patch.shape[0]
		int x = target.shape[1]
		int y = target.shape[0]
		int patchV, patchH

	patchV = int_min(i+patchSize, y) - i
	patchH = int_min(j+patchSize, x) - j
	target[i:int_min(i+patchSize, y), j:int_min(j+patchSize, x), :] = patch[:patchV, :patchH, :]

'''
This function takes in 2 overlapping image regions, computes pixel-wise L2 norm and returns cost map.
'''
cdef void makeCostMap(FLOAT[:,:,:] img1, FLOAT[:,:,:] img2, FLOAT[:,:] costMap) nogil:
	cdef:
		int i,j

	# calculate L2 norm distances of refPatch from patches
	for i in range(img1.shape[0]):
		for j in range(img1.shape[1]):
				costMap[i,j] = sqrt((img1[i,j,0] - img2[i,j,0])**2 + 
					(img1[i,j,1] - img2[i,j,1])**2 + (img1[i,j,2] - img2[i,j,2])**2)


'''
DP this shit, yo
'''
cdef void calcMinCosts(FLOAT[:,:] costMap) nogil:	
	cdef:
		# TODO does this copy or reference? - this needs to be a copy
		# FLOAT[:,:] cumuCosts = costMap
		# TODO: maybe try this: FLOAT[:,:] cumuCosts = costMap[:,:]
		int x = costMap.shape[1]
		int y = costMap.shape[0]
		FLOAT minVal
		int i, j

	for i in range(y - 1):
		for j in range(x):
			minVal = 99999.
			if j != 0 and minVal > costMap[i,j-1]:
				minVal = costMap[i,j-1]
			if j != x - 1 and minVal > costMap[i,j+1]:
				minVal = costMap[i,j+1]
			if minVal > costMap[i,j]:
				minVal = costMap[i,j]

			costMap[i + 1,j] += minVal


'''
Trace DP shit backwards, yo; "returning" pathCosts
'''
cdef void pathBacktrace(FLOAT[:,:] cumuCosts, INT[:,:] pathCosts) nogil:
	cdef:
		int x = cumuCosts.shape[1]
		int y = cumuCosts.shape[0]
		int minIdx, maxIdx, row, i, idx
		FLOAT minVal

	pathCosts[:,:] = 0

	minIdx = 0
	maxIdx = x - 1
	for row in range(y - 1, -1, -1):
		minVal = 999999.
		idx = 0
		# find index of minimum value in row (idx = np.argmin(cumuCosts[row, minIdx:maxIdx + 1])
		for i in range(minIdx, maxIdx + 1):
			if minVal > cumuCosts[row, i]:
				minVal = cumuCosts[row, i]
				idx = i
		pathCosts[row, idx] = 1

		# reset minIdx and maxIdx
		if idx - 1 > 0:
			minIdx = idx - 1
		else:
			minidx = 0

		if idx + 1 < x - 1:
			maxIdx = idx + 1
		else:
			maxIdx = x - 1


cdef void cheapVertPath(FLOAT[:,:] costMap, INT[:,:] pathCosts) nogil:
	cdef:
		# makes costMap cumulative in-place by DP
		calcMinCosts(costMap)

		# finds a path in the now cumulative costMap and marks it with 1's in pathCosts
		pathBacktrace(costMap, pathCosts)

'''
Generate binary mask
'''
cdef void cheapVertCut(FLOAT[:,:] costMap, INT[:,:] pathCosts) nogil:
	cdef:
		int row
		int x = pathCosts.shape[1]
		int y = pathCosts.shape[0]

	# fills in pathCosts with a path of 1's
	cheapVertPath(costMap, pathCosts)

	# fills in every single entry to the left of path with 1's
	for row in range(y):
		for col in range(x):
			if pathCosts[row, col] == 0:
				pathCosts[row, col] = 1
			else:
				break

# TODO oh god taking a transpose in C...
cdef void cheapHorizCut(FLOAT[:,:] costMap, INT[:,:] pathCosts) nogil:
	cdef:
		INT[:,:] path = cheapVertCut(costMap.T, pathCosts).T




























