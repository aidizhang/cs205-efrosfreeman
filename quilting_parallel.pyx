#cython: boundscheck=False, wraparound=False
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
		1) select randomly from candidate patches (within threshold_factor)
		2) calculate min error boundary
		3) patches into output texture
'''
cpdef void paste_patch(int texture_width, int texture_height, int tile_size,
					  int overlap, int num_rows, int num_cols, int tid,
					  FLOAT[:,:,:] np_texture, FLOAT[:,:,:,:] patches,
					  FLOAT[:,:,:] initial_patch):
	cdef:
		float threshold_factor = 1.1
		int i, j, chosen_idx
		int num_patches = patches.shape[0]
		int row_no = tid / num_cols
		int col_no = tid % num_cols
		int block_left, block_up
		int patch_size = overlap + tile_size
		int row_off, col_off

	# numpy arrays to store values made in calculations in cdef nogil functions
	np_d = np.zeros(patches.shape[0], dtype=np.float32)
	np_dist_left = np.zeros(patches.shape[0], dtype=np.float32)
	np_dist_up = np.zeros(patches.shape[0], dtype=np.float32)
	np_dist_both = np.zeros(patches.shape[0], dtype=np.float32)
	np_distances = np.empty_like(patches, dtype=np.float32)
	np_path_costs_left = np.zeros((patch_size, overlap), dtype=np.int32)
	np_path_costs_up = np.zeros((overlap, patch_size), dtype=np.int32)
	np_cost_map_left = np.zeros((patch_size, overlap), dtype=np.float32)
	np_cost_map_up = np.zeros((overlap, patch_size), dtype=np.float32)

	cdef:
		# typed MemoryViews on the numpy arrays
		FLOAT[:] d = np_d
		FLOAT[:] dist_left = np_dist_left
		FLOAT[:] dist_up = np_dist_up
		FLOAT[:] dist_both = np_dist_both
		FLOAT[:,:,:,:] distances = np_distances
		INT[:,:] path_costs_left = np_path_costs_left
		INT[:,:] path_costs_up = np_path_costs_up
		FLOAT[:,:] cost_map_left = np_cost_map_left
		FLOAT[:,:] cost_map_up = np_cost_map_up
		FLOAT[:,:,:] texture = np_texture
		# MemoryView on slices to be made
		FLOAT[:,:,:] ref_patch_left, ref_patch_up, ref_patch_both
		FLOAT[:,:,:] chosen_patch
	
	with nogil:
		block_left = 1 if col_no > 0 else 0
		block_up = 1 if row_no > 0 else 0
		
		# set offset in row and column, measured in pixels
		row_off = row_no * tile_size
		col_off = col_no * tile_size

		# find reference patchs and calculate overlap distances over all sample patches
		if block_left:
			ref_patch_left = texture[row_off:int_min(row_off + patch_size, texture_height), 
									 col_off:int_min(col_off + overlap, texture_width), :]
			overlap_distances(ref_patch_left, patches, dist_left)
			# alias to be used if patch bordered on two sides
			d = dist_left

		if block_up:
			ref_patch_up = texture[row_off:int_min(row_off + overlap, texture_height), 
								   col_off:int_min(col_off + patch_size, texture_width), :]
			overlap_distances(ref_patch_up, patches, dist_up)
			# alias to be used if patch bordered on two sides
			d = dist_up

		if block_left and block_up:
			ref_patch_both = texture[row_off:int_min(row_off + overlap, texture_height), 
									 col_off:int_min(col_off + overlap, texture_width), :]
			overlap_distances(ref_patch_both, patches, dist_both)
			# correct for overcounting in dist_both
			for i in range(num_patches):
				d[i] = dist_left[i] + dist_up[i] - dist_both[i]

		# finds appropriate random patch
		chosen_idx = get_matching_patch(d, threshold_factor)
		chosen_patch = patches[chosen_idx, :, :, :]

		# determines minimum cut boundary and overlays onto chosen patch
		if block_left:
			make_cost_map(ref_patch_left, chosen_patch[:ref_patch_left.shape[0], :overlap, :],
						cost_map_left)
			cheap_vert_cut(cost_map_left, path_costs_left)
			combine_ref_chosen(path_costs_left, ref_patch_left, chosen_patch)

		if block_up:
			make_cost_map(ref_patch_up, chosen_patch[:overlap, :ref_patch_up.shape[1], :],
						cost_map_up)
			cheap_horiz_cut(cost_map_up, path_costs_up)
			combine_ref_chosen(path_costs_up, ref_patch_up, chosen_patch)

		insert(texture, chosen_patch, row_off, col_off)

'''
int_min(a, b)
	returns the minimum of the integers a and b
'''
cdef inline int int_min(int a, int b) nogil: 
	return a if a <= b else b


'''
overlap_distances(ref_patch, patches, results)
	computes the L2 norm distance of ref_patch to each patch in patches
	stores total distance of each patch in results
'''
cdef void overlap_distances(FLOAT[:,:,:] ref_patch,
					   FLOAT[:,:,:,:] patches,
					   FLOAT[:] results) nogil:
	cdef:
		int num_patches = patches.shape[0]
		int height = ref_patch.shape[0]
		int width = ref_patch.shape[1]
		int i, j, k, p
	
	# calculate distances of ref_patch from patches
	# TODO: try different numbers of threads here
	# TODO: try load balancing with chunksize
	for i in prange(num_patches, num_threads=8, schedule='dynamic'):
		for j in range(height):
			for k in range(width):
				results[i] += sqrt((patches[i, j, k, 0] - ref_patch[j, k, 0])**2 + \
								   (patches[i, j, k, 1] - ref_patch[j, k, 1])**2 + \
								   (patches[i, j, k, 2] - ref_patch[j, k, 2])**2)


'''
combine_ref_chosen(path_mask, ref_patch, chosen_patch)
	using path_mask, copies pixels from ref_patch to chosen_path
	direction of iteration depends on dir: 0 for vertical overlap, 1 for horizontal
'''
cdef void combine_ref_chosen(INT[:,:] path_mask, 
						FLOAT[:,:,:] ref_patch, 
						FLOAT[:,:,:] chosen_patch) nogil:
	cdef:
		int width = ref_patch.shape[1]
		int height = ref_patch.shape[0]

	for i in range(height):
		for j in range(width):
			# use ref_patch if 1; chosen_patch if 0
			if path_mask[i][j] == 1:
				chosen_patch[i][j] = ref_patch[i][j]


'''
make_patches(img, patch_size)
	returns an array of all possible patches that can be made from img,
	with patch size patch_size
'''
def make_patches(img, patch_size):
	# check that img should have channel axis, so (x,y,channel)
	assert img.ndim == 3, "image should be RGB"

	nX = img.shape[0] - patch_size
	nY = img.shape[1] - patch_size
	num_chan = img.shape[2]
	patches = np.zeros((nX * nY, patch_size, patch_size, num_chan), dtype=np.float32)

	#iterate through all patches from img and store in patches
	k = 0
	for i in range(nX):
		for j in range(nY):
			x, X = i, i + patch_size
			y, Y = j, j + patch_size
			patches[k, :, :, :] = img[x:X, y:Y, :]
			k += 1

	return patches


'''
get_matching_patch(distances, threshold_factor)
	returns index of patch with error in distances,
	randomly selected from the set of patches with error within threshold_factor
	of the smallest error in distances (not including the exact matching patch)
'''
cdef int get_matching_patch(FLOAT[:] distances, float threshold_factor) nogil:
	cdef:
		FLOAT[:] d = distances
		int num_patches = distances.shape[0]
		int i
		float min_val

	# do not select exact matching patch
	min_val = 999999.
	for i in range(num_patches):
		# float imprecision
		if d[i] < min_val and d[i] > 0.01:
			min_val = d[i]

	cdef:
		float threshold = threshold_factor * min_val
		int ctr = 0

	# count number of qualifying indices to allocate memory for indices
	for i in range(num_patches):
		if d[i] < threshold:
			ctr += 1

	# allocate memory to store indices of candidate patches
	cdef:
		int* indices = <int *> malloc(ctr * sizeof(int))
		int temp = ctr

	# store all qualifying indices of d in indices
	for i in range(num_patches):
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
		int patch_idx = indices[idx]

	free(indices)

	return patch_idx


'''
insert(target, patch, i, j)
	copies patch into target at pixel position (i,j).
'''
cpdef void insert(FLOAT[:,:,:] target, FLOAT[:,:,:] patch, int i, int j) nogil:
	cdef:
		int patch_size = patch.shape[0]
		int x = target.shape[1]
		int y = target.shape[0]
		int patchV, patchH

	patchV = int_min(i + patch_size, y) - i
	patchH = int_min(j + patch_size, x) - j
	target[i:int_min(i + patch_size, y), j:int_min(j + patch_size, x), :] = patch[:patchV, :patchH, :]


'''
make_cost_map(img1, img2, cost_map)
	computes error pixel-wise by L2 norm and stores in cost_map
	img1 and img2 should have the same shape, and have 3 channels (RGB)
'''
cdef void make_cost_map(FLOAT[:,:,:] img1, FLOAT[:,:,:] img2, FLOAT[:,:] cost_map) nogil:
	cdef:
		int i,j
		int height = int_min(img1.shape[0], img2.shape[0])
		int width = int_min(img1.shape[1], img2.shape[1])

	# default error for the partial cost maps
	cost_map[:,:] = 999999.

	# calculate L2 norm distances of ref_patch from patches
	for i in range(height):
		for j in range(width):
				cost_map[i, j] = sqrt((img1[i, j, 0] - img2[i, j, 0])**2 + \
									 (img1[i, j, 1] - img2[i, j, 1])**2 + \
									 (img1[i, j, 2] - img2[i, j, 2])**2)


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
cdef void calc_min_costs(FLOAT[:,:] cost_map) nogil:	
	cdef:
		int x = cost_map.shape[1]
		int y = cost_map.shape[0]
		FLOAT min_val
		int i, j

	for i in range(y - 1):
		for j in range(x):
			min_val = 99999.
			# finds the min of its upper neighbors to the right, middle, and left
			if j != 0 and min_val > cost_map[i,j-1]:
				min_val = cost_map[i,j-1]
			if j != x - 1 and min_val > cost_map[i,j+1]:
				min_val = cost_map[i,j+1]
			if min_val > cost_map[i,j]:
				min_val = cost_map[i,j]

			cost_map[i + 1,j] += min_val


'''
path_backtrace(cumu_costs, path_costs)
	backtraces in the DP table cumu_costs and draws a path using 1's in path_costs
'''
cdef void path_backtrace(FLOAT[:,:] cumu_costs, INT[:,:] path_costs) nogil:
	cdef:
		int x = cumu_costs.shape[1]
		int y = cumu_costs.shape[0]
		int min_idx, max_idx, row, i, idx
		FLOAT min_val

	# default to 0
	path_costs[:,:] = 0

	# min_idx and max_idx will be updated so that we only search over neighboring
	# pixels in each previous row
	min_idx = 0
	max_idx = x - 1
	# iterate backwards through the DP table
	for row in range(y - 1, -1, -1):
		min_val = 999999.
		idx = 0
		# find index of minimum value in row
		for i in range(min_idx, max_idx + 1):
			if min_val > cumu_costs[row, i]:
				min_val = cumu_costs[row, i]
				idx = i
		path_costs[row, idx] = 1

		# reset min_idx and max_idx so that they don't go out of bounds
		min_idx = idx - 1 if idx - 1 > 0 else 0
		max_idx = int_min(idx + 1, x - 1)


'''
cheap_vert_path(cost_map, path_costs)
	calculates the min cost error boundary and stores it as 1's and 0's in path_costs
'''
cdef void cheap_vert_path(FLOAT[:,:] cost_map, INT[:,:] path_costs) nogil:
	# makes cost_map cumulative in-place by DP
	calc_min_costs(cost_map)

	# finds a path in the now cumulative cost_map and marks it with 1's in path_costs
	path_backtrace(cost_map, path_costs)


'''
cheap_vert_cut(cost_map, path_costs)
	creates a binary mask by setting everything to the left of the path
	to 1, signifying that those should be pixels from the existing texture
'''
cdef void cheap_vert_cut(FLOAT[:,:] cost_map, INT[:,:] path_costs) nogil:
	cdef:
		int row
		int x = path_costs.shape[1]
		int y = path_costs.shape[0]

	# fills in path_costs with a path of 1's
	cheap_vert_path(cost_map, path_costs)

	# fills in every single entry to the left of path with 1's
	for row in range(y):
		for col in range(x):
			if path_costs[row, col] == 0:
				path_costs[row, col] = 1
			else:
				break


'''
calc_min_costs_horiz(cost_map)
	same as vertical version, transposed
'''
cdef void calc_min_costs_horiz(FLOAT[:,:] cost_map) nogil:	
	cdef:
		int x = cost_map.shape[1]
		int y = cost_map.shape[0]
		FLOAT min_val
		int i, j

	for i in range(x - 1):
		for j in range(y):
			min_val = 99999.
			if j != 0 and min_val > cost_map[j - 1, i]:
				min_val = cost_map[j - 1, i]
			if j != y - 1 and min_val > cost_map[j + 1,i]:
				min_val = cost_map[j + 1, i]
			if min_val > cost_map[j, i]:
				min_val = cost_map[j, i]

			cost_map[j,i + 1] += min_val


'''
path_bactrace_horiz(cumu_costs, path_costs)
	same as vertical version, transposed
'''
cdef void path_backtrace_horiz(FLOAT[:,:] cumu_costs, INT[:,:] path_costs) nogil:
	cdef:
		int x = cumu_costs.shape[1]
		int y = cumu_costs.shape[0]
		int min_idx, max_idx, row, i, idx
		FLOAT min_val

	path_costs[:,:] = 0

	min_idx = 0
	max_idx = y - 1
	for col in range(x - 1, -1, -1):
		min_val = 999999.
		idx = 0
		for i in range(min_idx, max_idx + 1):
			if min_val > cumu_costs[i, col]:
				min_val = cumu_costs[i, col]
				idx = i
		path_costs[idx, col] = 1

		min_idx = idx - 1 if idx - 1 > 0 else 0
		max_idx = int_min(idx + 1, y - 1)


'''
cheap_horiz_path(cost_map, path_costs)
	same as vertical version, transposed
'''
cdef void cheap_horiz_path(FLOAT[:,:] cost_map, INT[:,:] path_costs) nogil:
	calc_min_costs_horiz(cost_map)
	path_backtrace_horiz(cost_map, path_costs)


'''
cheap_horiz_cut(cost_map, path_costs)
	same as vertical version, transposed
'''
cdef void cheap_horiz_cut(FLOAT[:,:] cost_map, INT[:,:] path_costs) nogil:
	cdef:
		int row
		int x = path_costs.shape[1]
		int y = path_costs.shape[0]

	cheap_horiz_path(cost_map, path_costs)

	for row in range(x):
		for col in range(y):
			if path_costs[col, row] == 0:
				path_costs[col, row] = 1
			else:
				break

