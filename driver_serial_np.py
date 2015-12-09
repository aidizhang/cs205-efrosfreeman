#!/usr/bin/python
'''
Driver for serial impementation of the Efros-Freeman image quilting algorithm,
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
import sys
import os
import random

from PIL import Image

from quilting_serial_np import *

if __name__ == "__main__":
	# read in original image using Python Image Library (PIL)
	image_name = "pebbles"
	orig_img = Image.open(image_name + ".png")
	(width, height) = orig_img.size

	# extract list of pixels in RGB/grayscale format
	pixels = list(orig_img.getdata())
	sample_2d = np.array(pixels, np.int32)
	sample_2d = sample_2d.reshape((height, -1, 3))

	# ensure that img is an RGB image
	assert sample_2d.ndim == 3 and sample_2d.shape[2] == 3, "input sample must be RGB"

	# choose patch from input sample by slicing
	patch_size = 30

	# define texture_size, tile_size and initialize blank canvas
	texture_size = (width * 2, height * 2)
	texture_width = texture_size[0]
	texture_height = texture_size[1]
	overlap = patch_size / 6
	tile_size = patch_size - overlap
	texture = np.zeros((texture_height, texture_width, 3), dtype=np.float32)

	# generate all sample patches
	patches = make_patches(sample_2d, patch_size)
	num_patches = patches.shape[0]

	# randomly select initial patch
	initial_patch = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
	rand_patch = random.randint(0, num_patches - 1)
	initial_patch[:,:,:] = patches[rand_patch,:,:,:]
	
	N = int(math.ceil(texture_width / float(tile_size)))
	M = int(math.ceil(texture_height / float(tile_size)))
	k = -1

	for i in range(M):
		for j in range(N):
			k += 1

			# insert default initial top-left patch
			if k == 0:
				insert(texture, initial_patch, i, j)
				continue

			block_left = j > 0
			block_up = i > 0

			# find reference patchs and calculate overlap distances over all sample patches
			if block_left:
				ref_patch_left = texture[i*tile_size:min(i*tile_size + patch_size, texture_height), 
								j*tile_size:min(j*tile_size + overlap, texture_width), :]
				dist_left = overlap_distances(ref_patch_left, patches)
				d = dist_left

			if block_up:
				ref_patch_up = texture[i*tile_size:min(i*tile_size + overlap, texture_height), 
								j*tile_size:min(j*tile_size + patch_size, texture_width), :]
				dist_up = overlap_distances(ref_patch_up, patches)
				d = dist_up

			if block_left and block_up:
				ref_patch_both = texture[i*tile_size:min(i*tile_size + overlap, texture_height), 
								j*tile_size:min(j*tile_size + overlap, texture_width), :]
				dist_both = overlap_distances(ref_patch_both, patches)
				d = dist_left + dist_up - dist_both

			# finds appropriate random patch
			chosen_idx = get_matching_patch(d, 1.1)
			chosen_patch = patches[chosen_idx, :, :, :]

			if block_left:
				cost_map = make_cost_map(ref_patch_left, chosen_patch[:ref_patch_left.shape[0], :overlap, :])
				path_mask_left = cheap_vert_cut(cost_map)
				overlap_left = np.where(np.dstack([path_mask_left] * 3), ref_patch_left,
									   			chosen_patch[:ref_patch_left.shape[0], :overlap, :])
				# overwrite with min cut
				chosen_patch[:ref_patch_left.shape[0], :overlap, :] = overlap_left

			if block_up:
				cost_map = make_cost_map(ref_patch_up, chosen_patch[:overlap, :ref_patch_up.shape[1], :])
				path_mask_up = cheap_horiz_cut(cost_map)
				overlap_up = np.where(np.dstack([path_mask_up] * 3), ref_patch_up,
										chosen_patch[:overlap, :ref_patch_up.shape[1], :])
				# overwrite with min cut
				chosen_patch[:overlap, :ref_patch_up.shape[1], :] = overlap_up

			if block_left and block_up:
				path_mask_both = np.zeros((ref_patch_up.shape[0], ref_patch_left.shape[1]))
				for p in range(ref_patch_up.shape[0]):
					for q in range(ref_patch_left.shape[1]):
						path_mask_both[p][q] = 1 - ((1 - path_mask_up[p, q]) * (1 - path_mask_left[p, q]))

				path_mask_left[:path_mask_both.shape[0], :] = path_mask_both
				path_mask_up[:,:path_mask_both.shape[1]] = path_mask_both

				overlap_both_left = np.where(np.dstack([path_mask_left] * 3), ref_patch_left,
												chosen_patch[:ref_patch_left.shape[0], :overlap, :])
				overlap_both_up = np.where(np.dstack([path_mask_up] * 3), ref_patch_up,
												chosen_patch[:overlap, :ref_patch_up.shape[1], :])
				
				# overwrite with min cut
				chosen_patch[:ref_patch_left.shape[0],:overlap,:] = overlap_both_left
				chosen_patch[:overlap,:ref_patch_up.shape[1],:] = overlap_both_up

			insert(texture, chosen_patch, i * tile_size, j * tile_size)
			print "Finished patch %i" % k

	# convert texture into flattened array pixels_out for exporting as PNG
	pixels_out = np.reshape(texture, (texture_width * texture_height, 3), order='C')
	pixels_out = map(lambda x: (x[0], x[1], x[2]), pixels_out)
	img_out = Image.new(orig_img.mode, texture_size)
	img_out.putdata(pixels_out)
	img_out.save(image_name + "_generated_" + str(patch_size) + ".png", "png")
	img_out.show()
	print "\nDone!\n"

