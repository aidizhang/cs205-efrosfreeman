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

# TODO: look at Liang paper


'''
schedule_paste_patch(texture, patches, initialPatch, metadata, tid, num_threads, events)
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
def schedulePastePatch(texture, patches, initialPatch, metadata, tid, num_threads, events):
	numRows = metadata['numRows']
	numCols = metadata['numCols']
	
	# the first one is already done
	if tid == 0:
		events[tid].set()
		sys.exit(0)

	# for each patch, make sure that left and up patches (if any) are done

	# if patch is on top boundary of texture
	if tid < numCols:
		events[tid - 1].wait()
	# if patch is on left boundary of texture
	elif tid % numCols == 0:
		events[tid - numCols].wait()
		events[tid - numCols + 1].wait()
	# if patch is on right boundary of texture
	elif tid % numCols == numCols - 1:
		events[tid - numCols].wait()
		events[tid - numCols - 1].wait()
		events[tid - 1].wait()
	# if patch is neither on top or left boundary of texture
	else:
		events[tid - numCols].wait()
		events[tid - numCols - 1].wait()
		events[tid - numCols + 1].wait()
		events[tid - 1].wait()

	# do work once ready
	pastePatch(metadata['textureWidth'],
			metadata['textureHeight'],
			metadata['tileSize'],
			metadata['overlap'],
			metadata['numRows'],
			metadata['numCols'],
			tid,
			texture,
			patches,
			initialPatch)

	# mark patch as done!
	events[tid].set()


'''
parallel_paste_patch(texture, patches, initialPatch, metadata)
	spawns threads, one for each patch to be inserted into the generated texture
	each thread works on schedule_paste_patch
'''
def parallelPastePatch(texture, patches, initialPatch, metadata):
	# keep track of finished patches in 2D events array
	numThreads = metadata['numCols'] * metadata['numRows']
	events = [threading.Event() for i in range(numThreads)]

	# spawn threads
	threads = []
	for tid in range(numThreads):
		threads.append(threading.Thread(target=schedulePastePatch, 
										args=(texture, patches, initialPatch,
											  metadata, tid, numThreads, events)))
		threads[tid].start()

	# finish threads
	for thread in threads:
		thread.join()

	return texture


if __name__ == "__main__":
	# read in original image using Python Image Library (PIL)
	orig_img = Image.open("basket.png")
	(width, height) = orig_img.size

	# extract list of pixels in RGB/grayscale format
	pixels = list(orig_img.getdata())
	sample_2d = np.array(pixels, np.int32)
	# TODO hardcoded constants
	sample_2d = sample_2d.reshape((height,-1,3))

	# ensure that img is an RGB image
	assert sample_2d.ndim == 3 and sample_2d.shape[2] == 3, sample_2d.shape

	# manually set patch_size
	# TODO hardcoded constants
	patchSize = 30

	# generate all sample patches
	patches = makePatches(sample_2d, patchSize)
	num_patches = patches.shape[0]

	# randomly select initial patch
	initialPatch = np.zeros((patchSize, patchSize, 3), dtype=np.float32)
	rand_patch = random.randint(0, num_patches - 1)
	initialPatch[:,:,:] = patches[rand_patch,:,:,:]

	# define textureSize, tileSize and initialize blank canvas
	# TODO hardcoded constants
	textureSize = (width * 2, height * 2)
	texture_width = textureSize[0]
	texture_height = textureSize[1]
	overlap = patchSize / 6
	tileSize = patchSize - overlap
	texture = np.zeros((texture_height, texture_width, 3), dtype=np.float32)
	
	# dimensions of patch grid needed to generated texture
	N = int(math.ceil(texture_width / float(tileSize)))
	M = int(math.ceil(texture_height / float(tileSize)))

	# create metadata for patch synthesis
	metadata = {'textureWidth':textureSize[0],
				'textureHeight':textureSize[1], 
				'overlap':overlap,
				'tileSize':tileSize,
				'numCols':N,
				'numRows':M}

	# insert initial seed patch into target
	insert(texture, initialPatch, 0, 0)

	# paste all patches in parallel, scheduling with condition variables
	parallelPastePatch(texture, patches, initialPatch, metadata)

	# convert texture into flattened array pixels_out for exporting as PNG
	pixels_out = np.reshape(texture, (texture_width * texture_height, 3), order='C')
	pixels_out = map(lambda x: (x[0], x[1], x[2]), pixels_out)
	img_out = Image.new(orig_img.mode, textureSize)
	img_out.putdata(pixels_out)
	# TODO save img_out
	img_out.show()
	print "\ndonedonedone!\n"
