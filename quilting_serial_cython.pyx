#cython: boundscheck=True, wraparound=False

cimport numpy as np
import numpy as np
cimport cython
from libc.math cimport sqrt
from libc.stdlib cimport rand, malloc, free

# numpy types
# TODO need all?
ctypedef np.float32_t FLOAT
ctypedef np.uint32_t UINT
ctypedef np.int32_t INT


# TODO: is this really necessary? just use python min in nogil
cdef inline int int_min(int a, int b) nogil: 
	return a if a <= b else b

'''
This function computes the distance of refPatch to all patches in patches, returning a 1D array of all distances.
Returns 1-D array of distances over all patches.
'''
# TODO: should make contiguous?
cpdef void overlapDistances(FLOAT[:,:,:] refPatch,
					   FLOAT[:,:,:,:] patches,
					   FLOAT[:,:,:,:] distances,
					   FLOAT[:] results) nogil:
	cdef:
		int numPatches = patches.shape[0]
		int i, j, k, p
	
	# calculate distances of refPatch from patches
	for i in range(numPatches):
		for j in range(refPatch.shape[0]):
			for k in range(refPatch.shape[1]):
				results[i] += sqrt((patches[i,j,k,0]-refPatch[j,k,0])**2 + (patches[i,j,k,1]-refPatch[j,k,1])**2 + (patches[i,j,k,2]-refPatch[j,k,2])**2)


cpdef void combineRefAndChosen(INT[:,:] pathMask, 
						FLOAT[:,:,:] refPatch, 
						FLOAT[:,:,:] chosenPatch, 
						int dir,
						int overlap, int tid) nogil:
	with gil:
		print "started combining ref and chosen for thread %i" % tid

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

	with gil:
		print "finished combining ref and chosen for thread %i" % tid


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


cpdef int getMatchingPatch(FLOAT[:] distances, float thresholdFactor, int tid) nogil:
	'''
	Given a 1-D array of patch distances, choose matching patch index that is within threshold.
	'''
	with gil:
		print "started finding matching patch for thread %i" % tid

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
	# threshold = thresholdFactor * minVal
	
	# count number of qualifying indices to allocate memory for indices
	for i in range(numPatches):
		if d[i] < threshold:
			# return i
			ctr += 1

	# with gil:
	# 	print "counter = number of qualifying indices: %i" % ctr

	cdef:
		int* indices = <int *> malloc(ctr * sizeof(int))
		int temp = ctr

	# store all qualifying indices of d in indices
	for i in range(numPatches):
		if d[i] < threshold:
			indices[temp - 1] = i
			temp -= 1
			if temp == 0:
				break
	
	cdef:
		int r = rand() # gives random number from 0 to RAND_MAX
		# TODO non-uniform
		int idx = r % ctr
		int patchIdx = indices[idx]

	free(indices)

	with gil:
		print "finished finding matching patch for thread %i" % tid

	return patchIdx

'''
This function inserts a patch into img at position (i,j).
'''
cpdef void insert(FLOAT[:,:,:] target, FLOAT[:,:,:] patch, int i, int j, int tid) nogil:
	with gil:
		print "started inserting for thread %i" % tid

	cdef:
		int patchSize = patch.shape[0]
		int x = target.shape[1]
		int y = target.shape[0]
		int patchV, patchH

	patchV = int_min(i+patchSize, y) - i
	patchH = int_min(j+patchSize, x) - j
	target[i:int_min(i+patchSize, y), j:int_min(j+patchSize, x), :] = patch[:patchV, :patchH, :]

	with gil:
		print "finished inserting for thread %i" % tid

'''
This function takes in 2 overlapping image regions, computes pixel-wise L2 norm and returns cost map.
'''
cpdef void makeCostMap(FLOAT[:,:,:] img1, FLOAT[:,:,:] img2, FLOAT[:,:] costMap, int tid) nogil:
	with gil:
		print "started making costmap for thread %i" % tid

	cdef:
		int i,j

	# calculate L2 norm distances of refPatch from patches
	for i in range(img1.shape[0]):
		for j in range(img1.shape[1]):
				costMap[i,j] = sqrt((img1[i,j,0] - img2[i,j,0])**2 + 
					(img1[i,j,1] - img2[i,j,1])**2 + (img1[i,j,2] - img2[i,j,2])**2)

	with gil:
		print "finished making costmap for thread %i" % tid

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
	# makes costMap cumulative in-place by DP
	calcMinCosts(costMap)

	# finds a path in the now cumulative costMap and marks it with 1's in pathCosts
	pathBacktrace(costMap, pathCosts)

'''
Generate binary mask
'''
cpdef void cheapVertCut(FLOAT[:,:] costMap, INT[:,:] pathCosts, int tid) nogil:
	with gil:
		print "started cheap vert cut for thread %i" % tid

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

	with gil:
		print "finished cheap vert cut for thread %i" % tid

# TODO oh god taking a transpose in C...
# cdef void cheapHorizCut(FLOAT[:,:] costMap, INT[:,:] pathCosts) nogil:
# 	cheapVertCut(costMap.T, pathCosts.T)


'''
DP this shit, yo
'''
cdef void calcMinCostsHoriz(FLOAT[:,:] costMap) nogil:	
	cdef:
		# TODO does this copy or reference? - this needs to be a copy
		# FLOAT[:,:] cumuCosts = costMap
		# TODO: maybe try this: FLOAT[:,:] cumuCosts = costMap[:,:]
		int x = costMap.shape[1] # row
		int y = costMap.shape[0] # column
		FLOAT minVal
		int i, j

	for i in range(y - 1): # column up to and including the penultimate column
		for j in range(x): # row
			minVal = 99999.
			if j != 0 and minVal > costMap[j-1,i]: # costMap always indexes row then column
				minVal = costMap[j-1,i]
			if j != x - 1 and minVal > costMap[j+1,i]:
				minVal = costMap[j+1,i]
			if minVal > costMap[j,i]:
				minVal = costMap[j,i]

			costMap[j,i+1] += minVal


'''
Trace DP shit backwards, yo; "returning" pathCosts
'''
cdef void pathBacktraceHoriz(FLOAT[:,:] cumuCosts, INT[:,:] pathCosts) nogil:
	cdef:
		int x = cumuCosts.shape[0] # row
		int y = cumuCosts.shape[1] # column
		int minIdx, maxIdx, row, i, idx
		FLOAT minVal

	pathCosts[:,:] = 0

	minIdx = 0
	maxIdx = x - 1
	for col in range(y - 1, -1, -1):
		minVal = 999999.
		idx = 0
		# find index of minimum value in row (idx = np.argmin(cumuCosts[row, minIdx:maxIdx + 1])
		for i in range(minIdx, maxIdx + 1):
			if minVal > cumuCosts[i, col]:
				minVal = cumuCosts[i, col]
				idx = i
		pathCosts[idx, col] = 1

		# reset minIdx and maxIdx
		if idx - 1 > 0:
			minIdx = idx - 1
		else:
			minidx = 0

		if idx + 1 < x - 1:
			maxIdx = idx + 1
		else:
			maxIdx = x - 1


cdef void cheapHorizPath(FLOAT[:,:] costMap, INT[:,:] pathCosts) nogil:
	# makes costMap cumulative in-place by DP
	calcMinCostsHoriz(costMap)

	# finds a path in the now cumulative costMap and marks it with 1's in pathCosts
	pathBacktraceHoriz(costMap, pathCosts)

'''
Generate binary mask
'''
cpdef void cheapHorizCut(FLOAT[:,:] costMap, INT[:,:] pathCosts, int tid) nogil:
	with gil:
		print "started cheap horiz cut for thread %i" % tid

	cdef:
		int row
		int x = pathCosts.shape[0]
		int y = pathCosts.shape[1]

	# fills in pathCosts with a path of 1's
	cheapHorizPath(costMap, pathCosts)

	# fills in every single entry to the left of path with 1's
	for row in range(x):
		for col in range(y):
			if pathCosts[row, col] == 0:
				pathCosts[row, col] = 1
			else:
				break

	with gil:
		print "finished cheap horiz cut for thread %i" % tid





















