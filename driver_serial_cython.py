'''
Driver for serial Cython impementation of the Efros-Freeman image quilting algorithm

Authors:
Aidi Zhang
Samuel Cheng

Harvard CS205, fall 2015
Course Head: Prof. Thouis (Ray) Jones
Teaching Fellow: Kevin Chen

'''

import sys
import os.path
sys.path.append(os.path.join('..', 'util'))

import numpy as np
from PIL import Image
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

import math
import random

from quilting_serial_cython import *


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
	assert sample_2d.ndim == 3 and sample_2d.shape[2] == 3, sample_2d.shape

	# manually set patch_size
	patch_size = 30

	# generate all sample patches
	patches = make_patches(sample_2d, patch_size)
	num_patches = patches.shape[0]

	# randomly select initial patch
	initial_patch = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
	rand_patch = random.randint(0, num_patches - 1)
	initial_patch[:,:,:] = patches[rand_patch,:,:,:]

	# define texture_size, tile_size and initialize blank canvas
	texture_size = (width * 2, height * 2)
	texture_width = texture_size[0]
	texture_height = texture_size[1]
	overlap = patch_size / 6
	tile_size = patch_size - overlap
	texture = np.zeros((texture_height, texture_width, 3), dtype=np.float32)
	
	# dimensions of patch grid needed to generated texture
	N = int(math.ceil(texture_width / float(tile_size)))
	M = int(math.ceil(texture_height / float(tile_size)))

	# insert patches to target
	k = -1
	for row_no in range(M):
		for col_no in range(N):
			d = np.zeros(patches.shape[0], dtype=np.float32)
			dist_left = np.zeros(patches.shape[0], dtype=np.float32)
			dist_up = np.zeros(patches.shape[0], dtype=np.float32)
			dist_both = np.zeros(patches.shape[0], dtype=np.float32)
			distances = np.empty_like(patches, dtype=np.float32)
			path_costs_left = np.zeros((patch_size, overlap), dtype=np.int32)
			path_costs_up = np.zeros((overlap, patch_size), dtype=np.int32)
			cost_map_left = np.zeros((patch_size, overlap), dtype=np.float32)
			cost_map_up = np.zeros((overlap, patch_size), dtype=np.float32)

			k += 1

			# set offset in row and column, measured in pixels
			row_off = row_no * tile_size
			col_off = col_no * tile_size

			# insert default initial top-left patch
			if k == 0:
				insert(texture, initial_patch, row_off, col_off)
				continue

			block_left = 1 if col_no > 0 else 0
			block_up = 1 if row_no > 0 else 0
			
			# find reference patchs and calculate overlap distances over all sample patches
			if block_left:
				# TODO does using python min affect performance enough?
				ref_patch_left = texture[row_off:min(row_off + patch_size, texture_height), 
										 col_off:min(col_off + overlap, texture_width), :]
				overlap_distances(ref_patch_left, patches, dist_left)
				# reference or actual copy if d = dist_left + dist_up - dist_both didn't work
				d = dist_left

			if block_up:
				ref_patch_up = texture[row_off:min(row_off + overlap, texture_height), 
									   col_off:min(col_off + patch_size, texture_width), :]
				overlap_distances(ref_patch_up, patches, dist_up)
				d = dist_up

			if block_left and block_up:
				ref_patch_both = texture[row_off:min(row_off + overlap, texture_height), 
										 col_off:min(col_off + overlap, texture_width), :]
				overlap_distances(ref_patch_both, patches, dist_both)
				for i in range(num_patches):
					d[i] = dist_left[i] + dist_up[i] - dist_both[i]

			threshold_factor = 1.1

			# finds appropriate random patch
			chosen_idx = get_matching_patch(d, threshold_factor)
			chosen_patch = patches[chosen_idx, :, :, :]

			# determines minimum cut boundary and overlays onto chosen patch
			if block_left:
				make_cost_map(ref_patch_left, chosen_patch[:ref_patch_left.shape[0], :overlap, :], cost_map_left)
				cheap_vert_cut(cost_map_left, path_costs_left)
				combine_ref_chosen(path_costs_left, ref_patch_left, chosen_patch)

			if block_up:
				make_cost_map(ref_patch_up, chosen_patch[:overlap, :ref_patch_up.shape[1], :], cost_map_up)
				cheap_horiz_cut(cost_map_up, path_costs_up)
				combine_ref_chosen(path_costs_up, ref_patch_up, chosen_patch)

			insert(texture, chosen_patch, row_off, col_off)
			print "Finished patch %i" % k

	# convert texture into flattened array pixels_out for exporting as PNG
	pixels_out = np.reshape(texture, (texture_width * texture_height, 3), order='C')
	pixels_out = map(lambda x: (x[0], x[1], x[2]), pixels_out)
	img_out = Image.new(orig_img.mode, texture_size)
	img_out.putdata(pixels_out)
	img_out.save(image_name + "_generated_" + str(patch_size) + ".png", "png")
	img_out.show()
	print "\nDone!\n"
