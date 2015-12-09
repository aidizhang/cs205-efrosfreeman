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

	vertScaleFactor = 2
	horizScaleFactor = 2

	assert vertScaleFactor >= 1 and horizScaleFactor >= 1, "cannot scale down"

	# extract list of pixels in RGB/grayscale format
	pixels = list(orig_img.getdata())
	sample_2d = np.array(pixels, np.int32)
	sample_2d = sample_2d.reshape((height, -1, 3))

	# ensure that img is an RGB image
	assert sample_2d.ndim == 3 and sample_2d.shape[2] == 3, "input sample must be RGB"

	# choose patch from input sample by slicing
	# TODO hard coded
	patchSize = 30
	sl = (slice(0, patchSize), slice(0, patchSize), slice(0, 3))
	# TODO: randomly select initial patch
	initialPatch = sample_2d[sl[0], sl[1], sl[2]]

	# define textureSize, tileSize and initialize blank canvas
	textureSize = (width * horizScaleFactor, height * vertScaleFactor)
	overlap = patchSize / 6
	tileSize = patchSize - overlap
	texture = np.zeros((textureSize[1], textureSize[0], 3), dtype=np.float32)

	# generate all sample patches
	patches = makePatches(sample_2d, patchSize)
	
	N = int(math.ceil(textureSize[0] / float(tileSize)))
	M = int(math.ceil(textureSize[1] / float(tileSize)))
	k = -1

	for i in range(M):
		for j in range(N):
			k += 1

			# insert default initial top-left patch
			if k == 0:
				insert(texture, initialPatch, i, j)
				continue

			blockLeft = j > 0
			blockUp = i > 0

			# find reference patchs and calculate overlap distances over all sample patches
			if blockLeft:
				refPatchLeft = texture[i*tileSize:min(i*tileSize + patchSize, textureSize[1]), 
								j*tileSize:min(j*tileSize + overlap, textureSize[0]), :]
				distLeft = overlapDistances(refPatchLeft, patches)
				d = distLeft

			if blockUp:
				refPatchUp = texture[i*tileSize:min(i*tileSize + overlap, textureSize[1]), 
								j*tileSize:min(j*tileSize + patchSize, textureSize[0]), :]
				distUp = overlapDistances(refPatchUp, patches)
				d = distUp

			if blockLeft and blockUp:
				refPatchBoth = texture[i*tileSize:min(i*tileSize + overlap, textureSize[1]), 
								j*tileSize:min(j*tileSize + overlap, textureSize[0]), :]
				distBoth = overlapDistances(refPatchBoth, patches)
				d = distLeft + distUp - distBoth

			# finds appropriate random patch
			chosenIdx = getMatchingPatch(d, 1.1)
			chosenPatch = patches[chosenIdx, :, :, :]

			if blockLeft:
				costMap = makeCostMap(refPatchLeft, chosenPatch[:refPatchLeft.shape[0], :overlap, :])
				pathMaskLeft = cheapVertCut(costMap)
				overlapLeft = np.where(np.dstack([pathMaskLeft] * 3), refPatchLeft,
									   chosenPatch[:refPatchLeft.shape[0], :overlap, :])
				# overwrite with min cut
				chosenPatch[:refPatchLeft.shape[0], :overlap, :] = overlapLeft

			if blockUp:
				# chosenSize = min(j*tileSize + patchSize, textureSize[0]) - j*tileSize
				# TODO: stupid solution; find better one
				costMap = makeCostMap(refPatchUp, chosenPatch[:overlap, :refPatchUp.shape[1], :])
				pathMaskUp = cheapHorizCut(costMap)
				overlapUp = np.where(np.dstack([pathMaskUp] * 3), refPatchUp,
									 chosenPatch[:overlap, :refPatchUp.shape[1], :])
				# overwrite with min cut
				chosenPatch[:overlap, :refPatchUp.shape[1], :] = overlapUp

			if blockLeft and blockUp:
				pathMaskBoth = np.zeros((refPatchUp.shape[0], refPatchLeft.shape[1]))
				for p in range(refPatchUp.shape[0]):
					for q in range(refPatchLeft.shape[1]):
						pathMaskBoth[p][q] = 1 - ((1 - pathMaskUp[p, q]) * (1 - pathMaskLeft[p, q]))

				pathMaskLeft[:pathMaskBoth.shape[0], :] = pathMaskBoth
				pathMaskUp[:,:pathMaskBoth.shape[1]] = pathMaskBoth

				overlapBothLeft = np.where(np.dstack([pathMaskLeft] * 3), refPatchLeft,
										   chosenPatch[:refPatchLeft.shape[0], :overlap, :])
				overlapBothUp = np.where(np.dstack([pathMaskUp] * 3), refPatchUp,
										 chosenPatch[:overlap, :refPatchUp.shape[1], :])
				
				# overwrite with min cut
				chosenPatch[:refPatchLeft.shape[0],:overlap,:] = overlapBothLeft
				chosenPatch[:overlap,:refPatchUp.shape[1],:] = overlapBothUp

			insert(texture, chosenPatch, i * tileSize, j * tileSize)

	# convert texture into flattened array pixels_out for exporting as PNG
	pixels_out = np.reshape(texture, (textureSize[0] * textureSize[1], 3), order='C')
	pixels_out = map(lambda x: (x[0], x[1], x[2]), pixels_out)
	img_out = Image.new(orig_img.mode, textureSize)
	img_out.putdata(pixels_out)
	img_out.save(image_name + "_generated_" + str(patchSize) + ".png", "png")
	img_out.show()

	print "donedonedone!"

