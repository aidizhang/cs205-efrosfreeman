#cython: boundscheck=True, wraparound=False
'''
Helper functions for parallel impementation of the Efros-Freeman image quilting algorithm

Authors:
Aidi Zhang
Samuel Cheng

Harvard CS205, fall 2015
Course Head: Prof. Thouis (Ray) Jones
Teaching Fellow: Kevin Chen

'''

cimport numpy as np
import numpy as np
cimport cython
from cython.parallel import parallel, prange
from libc.math cimport sqrt
from libc.stdlib cimport rand, malloc, free

# numpy dtypes
ctypedef np.float32_t FLOAT
ctypedef np.int32_t INT

# TODO: should make contiguous memory?


'''
paste_patch(texture_width, texture_height, tile_size, overlap,
			num_rows, num_cols, tid, np_texture, patches, initial_patch)
	inserts a new patch into the generated texture:
		1) select randomly from candidate patches
		2) calculate min error boundary
		3) patches into output texture
'''
cpdef void pastePatch(int textureWidth, int textureHeight, int tileSize,
					  int overlap, int numRows, int numCols, int tid,
					  FLOAT[:,:,:] np_texture, FLOAT[:,:,:,:] patches,
					  FLOAT[:,:,:] initialPatch):
	cdef:
		int i, j, chosenIdx
		int numPatches = patches.shape[0]
		int rowNo = tid / numCols
		int colNo = tid % numCols
		int blockLeft, blockUp
		int patchSize = overlap + tileSize
		int row_off, col_off

	# numpy arrays to store values made in calculations in cdef nogil functions
	np_d = np.zeros(patches.shape[0], dtype=np.float32)
	np_distLeft = np.zeros(patches.shape[0], dtype=np.float32)
	np_distUp = np.zeros(patches.shape[0], dtype=np.float32)
	np_distBoth = np.zeros(patches.shape[0], dtype=np.float32)
	np_distances = np.empty_like(patches, dtype=np.float32)
	np_pathCostsLeft = np.zeros((tileSize, overlap), dtype=np.int32)
	np_pathCostsUp = np.zeros((overlap, tileSize), dtype=np.int32)
	np_pathMaskBoth = np.zeros((overlap, overlap), dtype=np.int32)
	np_costMapLeft = np.zeros((tileSize, overlap), dtype=np.float32)
	np_costMapUp = np.zeros((overlap, tileSize), dtype=np.float32)

	cdef:
		# typed MemoryViews on the numpy arrays
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
		FLOAT[:,:,:] texture = np_texture
		# MemoryView on slices to be made
		FLOAT[:,:,:] refPatchLeft, refPatchUp, refPatchBoth
		FLOAT[:,:,:] chosenPatch
	
	with nogil:
		blockLeft = 1 if colNo > 0 else 0
		blockUp = 1 if rowNo > 0 else 0
		
		# set offset in row and column, measured in pixels
		row_off = rowNo * tileSize
		col_off = colNo * tileSize

		# find reference patchs and calculate overlap distances over all sample patches
		if blockLeft:
			refPatchLeft = texture[row_off:int_min(row_off + patchSize, textureHeight), 
								   col_off:int_min(col_off + overlap, textureWidth), :]
			overlapDistances(refPatchLeft, patches, distLeft)
			# alias to be used if patch bordered on two sides
			d = distLeft

		if blockUp:
			refPatchUp = texture[row_off:int_min(row_off + overlap, textureHeight), 
								 col_off:int_min(col_off + patchSize, textureWidth), :]
			overlapDistances(refPatchUp, patches, distUp)
			# alias to be used if patch bordered on two sides
			d = distUp

		if blockLeft and blockUp:
			refPatchBoth = texture[row_off:int_min(row_off + overlap, textureHeight), 
							col_off:int_min(col_off + overlap, textureWidth), :]
			overlapDistances(refPatchBoth, patches, distBoth)
			# correct for overcounting in distBoth
			for i in range(numPatches):
				d[i] = distLeft[i] + distUp[i] - distBoth[i]

		# finds appropriate random patch
		# TODO manually set threshold_factor
		chosenIdx = getMatchingPatch(d, 1.1)
		chosenPatch = patches[chosenIdx, :, :, :]

		# determines minimum cut boundary and overlays onto chosen patch
		if blockLeft:
			makeCostMap(refPatchLeft, chosenPatch[:refPatchLeft.shape[0], :overlap, :],
						costMapLeft)
			cheapVertCut(costMapLeft, pathCostsLeft)
			combineRefAndChosen(pathCostsLeft, refPatchLeft, chosenPatch, 0, overlap)

		if blockUp:
			makeCostMap(refPatchUp, chosenPatch[:overlap, :refPatchUp.shape[1], :],
						costMapUp)
			cheapHorizCut(costMapUp, pathCostsUp)
			combineRefAndChosen(pathCostsUp, refPatchUp, chosenPatch, 1, overlap)

		# TODO: is this even necessary, given how we're doing combineRefAndChosen?
		if blockLeft and blockUp:
			for i in range(overlap):
				for j in range(overlap):
					# bitwise or
					pathMaskBoth[i,j] = 1 - ((1-pathCostsUp[i,j]) * (1-pathCostsLeft[i,j]))

			pathCostsLeft[:overlap,:] = pathMaskBoth
			pathCostsUp[:,:overlap] = pathMaskBoth

			combineRefAndChosen(pathCostsLeft, refPatchLeft, chosenPatch, 0, overlap)
			combineRefAndChosen(pathCostsUp, refPatchUp, chosenPatch, 1, overlap)

		insert(texture, chosenPatch, row_off, col_off)

# TODO: is this really necessary? just use python min in nogil
cdef inline int int_min(int a, int b) nogil: 
	return a if a <= b else b


'''
overlap_distances(ref_patch, patches, results)
	computes the L2 norm distance of ref_patch to each patch in patches
	stores total distance of each patch in results
'''
cdef void overlapDistances(FLOAT[:,:,:] refPatch,
					   FLOAT[:,:,:,:] patches,
					   FLOAT[:] results) nogil:
	cdef:
		int numPatches = patches.shape[0]
		int height = refPatch.shape[0]
		int width = refPatch.shape[1]
		int i, j, k, p
	
	# calculate distances of refPatch from patches
	# TODO: try different numbers of threads here
	# TODO: try load balancing with chunksize
	for i in prange(numPatches, num_threads=8, schedule='dynamic'):
		for j in range(height):
			for k in range(width):
				results[i] += sqrt((patches[i, j, k, 0] - refPatch[j, k, 0])**2 + \
								   (patches[i, j, k, 1] - refPatch[j, k, 1])**2 + \
								   (patches[i, j, k, 2] - refPatch[j, k, 2])**2)


'''
combine_ref_chosen(path_mask, ref_patch, chosen_patch, dir, overlap)
	using path_mask, copies pixels from ref_patch to chosen_path
	direction of iteration depends on dir: 0 for vertical overlap, 1 for horizontal
'''
cdef void combineRefAndChosen(INT[:,:] pathMask, 
						FLOAT[:,:,:] refPatch, 
						FLOAT[:,:,:] chosenPatch, 
						int dir,
						int overlap) nogil:
	cdef:
		int width = refPatch.shape[1]
		int height = refPatch.shape[0]

	if dir == 0:
		for i in range(height):
			for j in range(overlap):
				# use refPatch if 1; chosenPatch if 0
				if pathMask[i][j] == 1:
					chosenPatch[i][j] = refPatch[i][j]
	else:
		for i in range(overlap):
			for j in range(width):
				# use refPatch if 1; chosenPatch if 0
				if pathMask[i][j] == 1:
					chosenPatch[i][j] = refPatch[i][j]


'''
make_patches(img, patch_size)
	returns an array of all possible patches that can be made from img,
	with patch size patch_size
'''
def makePatches(img, patchSize):
	# check that img should have channel axis, so (x,y,channel)
	assert img.ndim == 3, "image should be RGB"

	nX = img.shape[0] - patchSize
	nY = img.shape[1] - patchSize
	nChannels = img.shape[2]
	patches = np.zeros((nX * nY, patchSize, patchSize, nChannels), dtype=np.float32)

	#iterate through all patches from img and store in patches
	k = 0
	for i in range(nX):
		for j in range(nY):
			x, X = i, i + patchSize
			y, Y = j, j + patchSize
			patches[k, :, :, :] = img[x:X, y:Y, :]
			k += 1

	return patches


'''
get_matching_patch(distances, threshold_factor)
	returns index of patch with error in distances,
	randomly selected from the set of patches with error within threshold_factor
	of the smallest error in distances (not including the exact matching patch)
'''
cdef int getMatchingPatch(FLOAT[:] distances, float thresholdFactor) nogil:
	cdef:
		FLOAT[:] d = distances
		int numPatches = distances.shape[0]
		int i
		float minVal

	# do not select exact matching patch
	minVal = 999999.
	for i in range(numPatches):
		# float imprecision
		if d[i] < minVal and d[i] > 0.01:
			minVal = d[i]

	cdef:
		float threshold = thresholdFactor * minVal
		int ctr = 0

	# count number of qualifying indices to allocate memory for indices
	for i in range(numPatches):
		if d[i] < threshold:
			ctr += 1

	# allocate memory to store indices of candidate patches
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
		# r is pseudo-randomly selected from range 0 to RAND_MAX
		int r = rand()
		# slightly non-uniform, but not too important
		int idx = r % ctr
		int patchIdx = indices[idx]

	free(indices)

	return patchIdx


'''
insert(target, patch, i, j)
	copies patch into target at pixel position (i,j).
'''
cpdef void insert(FLOAT[:,:,:] target, FLOAT[:,:,:] patch, int i, int j) nogil:
	cdef:
		int patchSize = patch.shape[0]
		int x = target.shape[1]
		int y = target.shape[0]
		int patchV, patchH

	patchV = int_min(i + patchSize, y) - i
	patchH = int_min(j + patchSize, x) - j
	target[i:int_min(i + patchSize, y), j:int_min(j + patchSize, x), :] = patch[:patchV, :patchH, :]


'''
make_cost_map(img1, img2, cost_map)
	computes error pixel-wise by L2 norm and stores in cost_map
	img1 and img2 should have the same shape, and have 3 channels (RGB)
'''
cdef void makeCostMap(FLOAT[:,:,:] img1, FLOAT[:,:,:] img2, FLOAT[:,:] costMap) nogil:
	cdef:
		int i,j
		int height = img1.shape[0]
		int width = img1.shape[1]

	# calculate L2 norm distances of refPatch from patches
	for i in range(height):
		for j in range(width):
				costMap[i, j] = sqrt((img1[i, j, 0] - img2[i, j, 0])**2 + \
									 (img1[i, j, 1] - img2[i, j, 1])**2 + \
									 (img1[i,j,2] - img2[i,j,2])**2)


'''
calc_min_costs(cost_map)
	builds a DP table using cost_map that finds the min cost path
	each pixel depends on the min of the cumulative costs of its neighbors
	on the previous row
			X   X   X
		| 	  \ | /
		|	    X   X   X
		V         \ | /
		            X
'''
cdef void calcMinCosts(FLOAT[:,:] costMap) nogil:	
	cdef:
		int x = costMap.shape[1]
		int y = costMap.shape[0]
		FLOAT minVal
		int i, j

	for i in range(y - 1):
		for j in range(x):
			minVal = 99999.
			# finds the min of its upper neighbors to the right, middle, and left
			if j != 0 and minVal > costMap[i,j-1]:
				minVal = costMap[i,j-1]
			if j != x - 1 and minVal > costMap[i,j+1]:
				minVal = costMap[i,j+1]
			if minVal > costMap[i,j]:
				minVal = costMap[i,j]

			costMap[i + 1,j] += minVal


'''
path_backtrace(cumu_costs, path_costs)
	backtraces in the DP table cumu_costs and draws a path using 1's in path_costs
'''
cdef void pathBacktrace(FLOAT[:,:] cumuCosts, INT[:,:] pathCosts) nogil:
	cdef:
		int x = cumuCosts.shape[1]
		int y = cumuCosts.shape[0]
		int minIdx, maxIdx, row, i, idx
		FLOAT minVal

	# default to 0
	pathCosts[:,:] = 0

	# min_idx and max_idx will be updated so that we only search over neighboring
	# pixels in each previous row
	minIdx = 0
	maxIdx = x - 1
	# iterate backwards through the DP table
	for row in range(y - 1, -1, -1):
		minVal = 999999.
		idx = 0
		# find index of minimum value in row
		for i in range(minIdx, maxIdx + 1):
			if minVal > cumuCosts[row, i]:
				minVal = cumuCosts[row, i]
				idx = i
		pathCosts[row, idx] = 1

		# TODO functional programming
		# reset minIdx and maxIdx
		if idx - 1 > 0:
			minIdx = idx - 1
		else:
			minidx = 0

		if idx + 1 < x - 1:
			maxIdx = idx + 1
		else:
			maxIdx = x - 1


'''
cheap_vert_path(cost_map, path_costs)
	calculates the min cost error boundary and stores it as 1's and 0's in path_costs
'''
cdef void cheapVertPath(FLOAT[:,:] costMap, INT[:,:] pathCosts) nogil:
	# makes costMap cumulative in-place by DP
	calcMinCosts(costMap)

	# finds a path in the now cumulative costMap and marks it with 1's in pathCosts
	pathBacktrace(costMap, pathCosts)


'''
cheap_vert_cut(cost_map, path_costs)
	creates a binary mask by setting everything to the left of the path
	to 1, signifying that those should be pixels from the existing texture
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


'''
calc_min_costs_horiz(cost_map)
	same as vertical version, transposed
'''
cdef void calcMinCostsHoriz(FLOAT[:,:] costMap) nogil:	
	cdef:
		int x = costMap.shape[1]
		int y = costMap.shape[0]
		FLOAT minVal
		int i, j

	for i in range(x - 1):
		for j in range(y):
			minVal = 99999.
			if j != 0 and minVal > costMap[j-1,i]:
				minVal = costMap[j-1,i]
			if j != y - 1 and minVal > costMap[j+1,i]:
				minVal = costMap[j+1,i]
			if minVal > costMap[j,i]:
				minVal = costMap[j,i]

			costMap[j,i+1] += minVal


'''
path_bactrace_horiz(cumu_costs, path_costs)
	same as vertical version, transposed
'''
cdef void pathBacktraceHoriz(FLOAT[:,:] cumuCosts, INT[:,:] pathCosts) nogil:
	cdef:
		int x = cumuCosts.shape[1]
		int y = cumuCosts.shape[0]
		int minIdx, maxIdx, row, i, idx
		FLOAT minVal

	pathCosts[:,:] = 0

	minIdx = 0
	maxIdx = y - 1
	for col in range(x - 1, -1, -1):
		minVal = 999999.
		idx = 0
		for i in range(minIdx, maxIdx + 1):
			if minVal > cumuCosts[i, col]:
				minVal = cumuCosts[i, col]
				idx = i
		pathCosts[idx, col] = 1

		if idx - 1 > 0:
			minIdx = idx - 1
		else:
			minidx = 0

		if idx + 1 < x - 1:
			maxIdx = idx + 1
		else:
			maxIdx = x - 1


'''
cheap_horiz_path(cost_map, path_costs)
	same as vertical version, transposed
'''
cdef void cheapHorizPath(FLOAT[:,:] costMap, INT[:,:] pathCosts) nogil:
	calcMinCostsHoriz(costMap)
	pathBacktraceHoriz(costMap, pathCosts)


'''
cheap_horiz_cut(cost_map, path_costs)
	same as vertical version, transposed
'''
cdef void cheapHorizCut(FLOAT[:,:] costMap, INT[:,:] pathCosts) nogil:
	cdef:
		int row
		int x = pathCosts.shape[1]
		int y = pathCosts.shape[0]

	cheapHorizPath(costMap, pathCosts)

	for row in range(x):
		for col in range(y):
			if pathCosts[col, row] == 0:
				pathCosts[col, row] = 1
			else:
				break

