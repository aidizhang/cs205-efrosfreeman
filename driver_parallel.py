'''
Driver for parallel impementation of the Efros-Freeman image quilting algorithm

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
import threading

from quilting_parallel import *


'''
schedule_paste_patch(texture, patches, initial_patch, metadata, tid, num_threads, events)
	controls the scheduling of threads to select patches, calculate error boundary,
	and insert them into the generated texture
	schedules such that each thread controls one patch, and it can start only when
	the patches to the left and up are done:

		A   B   C
		  \ | /
		D - E

	thread A (the first thread) does not need to wait (it is already inserted)
	thread B needs to wait on patch A to finish
	thread C needs to wait on patch B to finish
	thread D needs to wait on patches A and D to finish
	thread E needs to wait on patches A, B, C, and D to finish
'''
def schedule_paste_patch(texture, patches, initial_patch, metadata, tid, num_threads, events):
	num_rows = metadata['num_rows']
	num_cols = metadata['num_cols']
	
	# the first one is already done
	if tid == 0:
		events[tid].set()
		sys.exit(0)

	# for each patch, make sure that left and up patches (if any) are done

	# if patch is on top boundary of texture
	if tid < num_cols:
		events[tid - 1].wait()
	# if patch is on left boundary of texture
	elif tid % num_cols == 0:
		events[tid - num_cols].wait()
		events[tid - num_cols + 1].wait()
	# if patch is on right boundary of texture
	elif tid % num_cols == num_cols - 1:
		events[tid - num_cols].wait()
		events[tid - num_cols - 1].wait()
		events[tid - 1].wait()
	# if patch is neither on top or left boundary of texture
	else:
		events[tid - num_cols].wait()
		events[tid - num_cols - 1].wait()
		events[tid - num_cols + 1].wait()
		events[tid - 1].wait()

	# do work once ready
	paste_patch(metadata['texture_width'],
			metadata['texture_height'],
			metadata['tile_size'],
			metadata['overlap'],
			metadata['num_rows'],
			metadata['num_cols'],
			tid,
			texture,
			patches,
			initial_patch)

	# mark patch as done!
	events[tid].set()

	print "Finished patch %i" % tid


'''
parallel_paste_patch(texture, patches, initial_patch, metadata)
	spawns threads, one for each patch to be inserted into the generated texture
	each thread works on schedule_paste_patch
'''
def parallel_paste_patch(texture, patches, initial_patch, metadata):
	# keep track of finished patches in 2D events array
	num_threads = metadata['num_cols'] * metadata['num_rows']
	events = [threading.Event() for i in range(num_threads)]

	# spawn threads
	threads = []
	for tid in range(num_threads):
		threads.append(threading.Thread(target=schedule_paste_patch, 
										args=(texture, patches, initial_patch,
											  metadata, tid, num_threads, events)))
		threads[tid].start()

	# finish threads
	for thread in threads:
		thread.join()

	return texture


if __name__ == "__main__":
	# read in original image using Python Image Library (PIL)
	image_name = "pebbles"
	orig_img = Image.open(image_name + ".png")
	(width, height) = orig_img.size

	# extract list of pixels in RGB/grayscale format
	pixels = list(orig_img.getdata())
	sample_2d = np.array(pixels, np.int32)
	# TODO hardcoded constants
	sample_2d = sample_2d.reshape((height, -1, 3))

	# ensure that img is an RGB image
	assert sample_2d.ndim == 3 and sample_2d.shape[2] == 3, sample_2d.shape

	# manually set patch_size
	# TODO hardcoded constants
	patch_size = 30

	# generate all sample patches
	patches = make_patches(sample_2d, patch_size)
	num_patches = patches.shape[0]

	# randomly select initial patch
	initial_patch = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
	rand_patch = random.randint(0, num_patches - 1)
	initial_patch[:,:,:] = patches[rand_patch,:,:,:]

	# define texture_size, tileSize and initialize blank canvas
	# TODO hardcoded constants
	texture_size = (width * 2, height * 2)
	texture_width = texture_size[0]
	texture_height = texture_size[1]
	overlap = patch_size / 6
	tile_size = patch_size - overlap
	texture = np.zeros((texture_height, texture_width, 3), dtype=np.float32)
	
	# dimensions of patch grid needed to generated texture
	N = int(math.ceil(texture_width / float(tile_size)))
	M = int(math.ceil(texture_height / float(tile_size)))

	# create metadata for patch synthesis
	metadata = {'texture_width':texture_width,
				'texture_height':texture_height, 
				'overlap':overlap,
				'tile_size':tile_size,
				'num_cols':N,
				'num_rows':M}

	# insert initial seed patch into target
	insert(texture, initial_patch, 0, 0)

	# paste all patches in parallel, scheduling with condition variables
	parallel_paste_patch(texture, patches, initial_patch, metadata)

	# convert texture into flattened array pixels_out for exporting as PNG
	pixels_out = np.reshape(texture, (texture_width * texture_height, 3), order='C')
	pixels_out = map(lambda x: (x[0], x[1], x[2]), pixels_out)
	img_out = Image.new(orig_img.mode, texture_size)
	img_out.putdata(pixels_out)
	img_out.save(image_name + "_generated_" + str(patch_size) + ".png", "png")
	img_out.show()
	print "\nDone!\n"
