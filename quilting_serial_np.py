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
This function computes the distance of ref_patch to all patches in patches
Returns 1-D array of distances over all patches.
'''
def overlap_distances(ref_patch, patches):
	# find ref_patch area of each sample patch
	ov = patches[:, :ref_patch.shape[0], :ref_patch.shape[1], :]

	# find distances of ref_patch area of sample patches from ref_patch itself
	num_patches = patches.shape[0]
	distances = ov - np.tile(ref_patch, (num_patches, 1, 1, 1))

	# calculate L2 norm and sum over all reference patch pixels
	# TODO #1: parallelize
	distances = np.sqrt(np.sum(np.square(distances), axis=3))
	distances = np.sum(distances, axis=1)
	distances = np.sum(distances, axis=1)
	
	return distances


'''
This function takes in an img with size img.shape and a patch size patch_size,
returns an array of shape (patch_size, patch_size, #num of patches),
so (:,:,idx) returns the idx'th patch
'''
def make_patches(img, patch_size):
	#check that img should have channel axis, so (x,y,channel)
	assert img.ndim == 3, "image should have channel axis"

	nX = img.shape[0] - patch_size
	nY = img.shape[1] - patch_size
	num_chan = img.shape[2]
	patches = np.zeros((nX * nY, patch_size, patch_size, num_chan), img.dtype)

	#iterate through all patches from img and store in patches
	k = 0
	for i in range(nX):
		for j in range(nY):
			x,X = i, i + patch_size
			y,Y = j, j + patch_size
			patches[k, :, :, :] = img[x:X, y:Y, :]
			k += 1

	return patches


'''
Given a 1-D array of patch distances, choose matching patch index that is within threshold.
'''
def get_matching_patch(distances, threshold_factor):
	d = distances
	# do not select current patch
	d[d < np.finfo(d.dtype).eps] = 99999
	m = np.min(d)
	# choose random index such that the distance is within threshold factor of minimum distance
	# TODO: make default threshold_factor
	threshold = threshold_factor * m
	indices = np.where(d < threshold)[0]
	idx = indices[np.random.randint(0, len(indices))]
	return idx


'''
This function inserts a patch into img at position (i,j).
'''
def insert(target, patch, i, j):
	patch_size = patch.shape[0]
	patchV = min(i + patch_size, target.shape[0]) - i
	patchH = min(j + patch_size, target.shape[1]) - j
	target[i:min(i + patch_size, target.shape[0]), \
		   j:min(j + patch_size, target.shape[1]), :] = patch[:patchV, :patchH, :]


'''
This function takes in 2 overlapping image regions, computes pixel-wise L2 norm and returns cost map.
'''
def make_cost_map(img1, img2):
	return np.sqrt(np.sum(np.square(img1 - img2), axis=2))


'''
DP
'''
def calc_min_costs(cost_map):
	cumu_costs = np.ones(cost_map.shape)
	x = cost_map.shape[1]
	y = cost_map.shape[0]
	
	cumu_costs[:] = cost_map[:]
	for i in range(y - 1):
		for j in range(x):
			if j == 0:
				c = cumu_costs[i, 0:2]
			elif j == x - 1:
				c = cumu_costs[i, x - 2:x]
			else:
				c = cumu_costs[i, j - 1:j + 2]

			cumu_costs[i + 1, j] += np.min(c)

	return cumu_costs

'''
backtrace on DP matrix to find path
'''
def path_backtrace(cumu_costs):
	x = cumu_costs.shape[1]
	y = cumu_costs.shape[0]

	path_costs = np.zeros(cumu_costs.shape)

	min_idx = 0
	max_idx = x - 1
	for row in range(y - 1, -1, -1):
		i = np.argmin(cumu_costs[row, min_idx:max_idx + 1])
		path_costs[row, i] = 1
		min_idx = np.max([0, i - 1])
		max_idx = np.min([x - 1, i + 1])

	return path_costs


def cheap_vert_path(cost_map):
	costs = calc_min_costs(cost_map)
	path = path_backtrace(costs)
	return path


'''
Generate binary mask
'''
def cheap_vert_cut(cost_map):
	path = cheap_vert_path(cost_map)

	for row in range(path.shape[0]):
		path[row, 0:np.argmax(path[row, :])] = 1
	return path


def cheap_horiz_cut(cost_map):
	path = cheap_vert_cut(cost_map.T).T
	return path

