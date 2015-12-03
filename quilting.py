#!/usr/bin/python

import numpy as np
import math
import sys
import os
import png
import itertools
import random

from pylab import *
from PIL import Image
from numpngw import write_png
# from quilting_helpers import verticalPathsCost
# from quilting_helpers import calculateCost

def overlapDistances(refPatch, patches):
	'''
	This function computes the distance of refPatch to all patches in patches, returning a 1D array of all distances.
	Returns 1-D array of distances over all patches.
	'''
	# find refPatch area of each sample patch
	ov = patches[:, :refPatch.shape[0], :refPatch.shape[1], :]

	# find distances of refPatch area of sample patches from refPatch itself
	numPatches = patches.shape[0]
	distances = ov - np.tile(refPatch, (numPatches,1,1,1))

	# calculate L2 norm and sum over all reference patch pixels
	distances = np.sqrt(np.sum(np.square(distances), axis=3))
	distances = np.sum(distances, axis=1)
	distances = np.sum(distances, axis=1)
	
	return distances


def makePatches(img, patchSize):
	'''
	This function takes in an img with size img.shape and a patch size patchSize, returns an array of shape
	(patchSize, patchSize, #num of patches), so (:,:,idx) returns the idx'th patch
	'''
	#check that img should have channel axis, so (x,y,channel)
	assert img.ndim == 3, "image should have channel axis"

	nX = img.shape[0] - patchSize
	nY = img.shape[1] - patchSize
	nChannels = img.shape[2]
	patches = np.zeros((nX*nY, patchSize, patchSize, nChannels), img.dtype)

	#iterate through all patches from img and store in patches
	k = 0
	for i in range(nX):
		for j in range(nY):
			x,X = i, i+patchSize
			y,Y = j, j+patchSize
			patches[k,:,:,:] = img[x:X,y:Y,:]
			k += 1

	return patches


def getMatchingPatch(distances, thresholdFactor):
	'''
	Given a 1-D array of patch distances, choose matching patch index that is within threshold.
	'''
	d = distances
	# do not select current patch
	d[d < np.finfo(d.dtype).eps] = 99999
	m = np.min(d)
	# choose random index such that the distance is within threshold factor of minimum distance
	# TODO: make default thresholdFactor
	threshold = thresholdFactor * m
	indices = np.where(d < threshold)[0]
	idx = indices[np.random.randint(0,len(indices))]
	return idx


def mkTexture(textureSize, patches, overlap):
	'''
	Main function
	'''
	patchSize = patches.shape[0]
	nChannels = 3 # currently hardcoded
	tileSize = patchSize - overlap
	texture = np.zeros((textureSize[0], textureSize[1], nChannels), dtype=np.float32)

	nPatches = patches.shape[2]

	k = -1
	width, height = int(math.ceil(textureSize[0]/float(tileSize))), int(math.ceil(textureSize[0]/float(tileSize)))
	for i in range(width):
		for j in range(height):
			k += 1

			#use random patch as first patch
			texture[0:patchSize, 0:patchSize, :] = patches[:,:,:,np.random.randint(0,nPatches)]

			#slicing for left overlap
			sl_l = (slice(i*tileSize, min(i*tileSize + patchSize, texture.shape[0])), 
						slice(j*tileSize, min(j*tileSize + overlap, texture.shape[1])), slice(0, nChannels))

			#slicing for writing PATCH at position (i,j)
			sl_patch = (slice(i*tileSize, min(i*tileSize + patchSize, texture.shape[0])), 
						slice(j*tileSize, min(j*tileSize + patchSize, texture.shape[1])), slice(0, nChannels))

			#finds minimum overlap, and finds minimum distance to available patches
			ov1 = texture[sl_l[0], sl_l[1], :, np.newaxis]
			d = patchDistance(ov1, patches)

            #choose best possible matched patch
			chosenPatchIndex = getMatchingPatch(d)
			chosenPatch =  patches[sl_patch[0], sl_patch[1], :, chosenPatchIndex]

			#paste chosenPatch at texture at position (i,j)
			texture[sl_patch] = chosenPatch

	return texture


def insert(target, patch, i, j):
	'''
	This function inserts a patch into img at position (i,j).
	'''
	patchSize = patch.shape[0]
	patchV = min(i+patchSize, target.shape[0]) - i
	patchH = min(j+patchSize, target.shape[1]) - j
	target[i:min(i+patchSize, target.shape[0]), j:min(j+patchSize, target.shape[1]), :] = patch[:patchV, :patchH, :]


def makeCostMap(img1, img2):
	'''
	This function takes in 2 overlapping image regions, computes pixel-wise L2 norm and returns cost map.
	'''
	return np.sqrt(np.sum(np.square(img1-img2), axis=2))

def calcMinCosts(costMap):
	'''
	DP this shit, yo
	'''
	cumuCosts = np.ones(costMap.shape)
	x = costMap.shape[1]
	y = costMap.shape[0]
	
	cumuCosts[:] = costMap[:]
	for i in range(y - 1):
		for j in range(x):
			if j == 0:
				c = cumuCosts[i, 0:2]
			elif j == x - 1:
				c = cumuCosts[i, x - 2:x]
			else:
				c = cumuCosts[i, j - 1:j + 2]

			cumuCosts[i + 1,j] += np.min(c)

	return cumuCosts

def pathBacktrace(cumuCosts):
	'''
	trace DP shit backwards, yo
	'''
	x = cumuCosts.shape[1]
	y = cumuCosts.shape[0]

	pathCosts = np.zeros(cumuCosts.shape)

	minIdx = 0
	maxIdx = x - 1
	for row in range(y - 1, -1, -1):
		i = np.argmin(cumuCosts[row, minIdx:maxIdx + 1])
		pathCosts[row, i] = 1
		minIdx = np.max([0, i - 1])
		maxIdx = np.min([x - 1, i + 1])

	return pathCosts

def cheapVertPath(costMap):
	costs = calcMinCosts(costMap)
	path = pathBacktrace(costs)
	return path

def cheapVertCut(costMap):
	'''
	Generate binary mask
	'''
	path = cheapVertPath(costMap)

	for row in range(path.shape[0]):
		path[row, 0:np.argmax(path[row, :])] = 1
	return path


def cheapHorizCut(costMap):
	path = cheapVertCut(costMap.T).T
	return path


if __name__ == "__main__":
	# read in original image using Python Image Library (PIL)
	orig_img = Image.open("basket.png")
	(width, height) = orig_img.size

	# extract list of pixels in RGB/grayscale format
	pixels = list(orig_img.getdata())
	sample_2d = np.array(pixels, np.int32)
	sample_2d = sample_2d.reshape((height,-1,3))

	# ensure that img is either an RGB or grayscale image
	assert sample_2d.ndim == 3 and (sample_2d.shape[2] == 3 or sample_2d.shape[2] == 1), sample_2d.shape

	# choose patch from input sample by slicing
	patchSize = 30
	sl = (slice(0,patchSize), slice(0,patchSize), slice(0,3))
	# TODO: randomly select initial patch
	initialPatch = sample_2d[sl[0], sl[1], sl[2]]

	# define textureSize, tileSize and initialize blank canvas
	textureSize = (width * 2, height * 2)
	overlap = patchSize / 6
	tileSize = patchSize - overlap
	texture = np.zeros((textureSize[1], textureSize[0], 3), dtype=np.float32)

	# generate all sample patches
	patches = makePatches(sample_2d, 30)
	
	N = int(math.ceil(textureSize[0]/float(tileSize)))
	M = int(math.ceil(textureSize[1]/float(tileSize)))
	k = -1

	for i in range(M): # height M
		for j in range(N): # width N
			k += 1
			# insert default initial top-left patch
			if k == 0:
				insert(texture, initialPatch, i, j)
				continue

			blockLeft = j>0
			blockUp = i>0

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
				overlapLeft = np.where(np.dstack([pathMaskLeft] * 3), refPatchLeft, chosenPatch[:refPatchLeft.shape[0], :overlap, :])
				# overwrite with min cut
				chosenPatch[:refPatchLeft.shape[0],:overlap,:] = overlapLeft

			if blockUp:
				# chosenSize = min(j*tileSize + patchSize, textureSize[0]) - j*tileSize
				# TODO: stupid solution; find better one
				costMap = makeCostMap(refPatchUp, chosenPatch[:overlap, :refPatchUp.shape[1], :])
				pathMaskUp = cheapHorizCut(costMap)
				overlapUp = np.where(np.dstack([pathMaskUp] * 3), refPatchUp, chosenPatch[:overlap, :refPatchUp.shape[1], :])
				# overwrite with min cut
				chosenPatch[:overlap,:refPatchUp.shape[1],:] = overlapUp

			if blockLeft and blockUp:
				pathMaskBoth = np.zeros((refPatchUp.shape[0], refPatchLeft.shape[1]))
				for p in range(refPatchUp.shape[0]):
					for q in range(refPatchLeft.shape[1]):
						pathMaskBoth[p][q] = 1 - ((1-pathMaskUp[p][q]) * (1-pathMaskLeft[p][q]))
						# pathMaskBoth[p][q] = pathMaskUp[p][q] | pathMaskLeft[p][q]

				pathMaskLeft[:pathMaskBoth.shape[0],:] = pathMaskBoth
				pathMaskUp[:,:pathMaskBoth.shape[1]] = pathMaskBoth

				overlapBothLeft = np.where(np.dstack([pathMaskLeft] * 3), refPatchLeft, chosenPatch[:refPatchLeft.shape[0], :overlap, :])
				overlapBothUp = np.where(np.dstack([pathMaskUp] * 3), refPatchUp, chosenPatch[:overlap, :refPatchUp.shape[1], :])
				
				# overwrite with min cut
				chosenPatch[:refPatchLeft.shape[0],:overlap,:] = overlapBothLeft
				chosenPatch[:overlap,:refPatchUp.shape[1],:] = overlapBothUp

			insert(texture, chosenPatch, i*tileSize, j*tileSize)

	# convert texture into flattened array pixels_out for exporting as PNG
	pixels_out = np.reshape(texture, (textureSize[0] * textureSize[1], 3), order='C')
	pixels_out = map(lambda x: (x[0],x[1],x[2]), pixels_out)
	img_out = Image.new(orig_img.mode, textureSize)
	img_out.putdata(pixels_out)
	img_out.show()





























