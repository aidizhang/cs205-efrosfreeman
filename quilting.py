#!/usr/bin/python

import numpy as np
import math
import sys
import os
import png
import itertools
from pylab import *
# import matplotlib.pyplot as plt
# import vigra
from PIL import Image
from numpngw import write_png
from image_quilting_helpers import verticalPathsCost
from image_quilting_helpers import calculateCost

def patchDistance(refPatch, patches):
	'''
	This function computes the distance of refPatch to all patches in patches, returning a 1D array of all distances.
	'''
	ov = patches[:refPatch.shape[0], :refPatch.shape[1],:,:]
	distances = ov - np.tile(refPatch, (1,1,1,patches.shape[3]))
	distances = np.sqrt(np.sum(np.square(distances), axis=2))
	distances = np.sum(distances, axis=0)
	return distances

def mkPatches(img, patchSize):
	'''
	This function takes in an img with size img.shape and a patch size patchSize, returns an array of shape
	(patchSize, patchSize, #num of patches), so (:,:,idx) returns the idx'th patch
	'''
	#check that img should have channel axis, so (x,y,channel)
	assert img.ndim == 3, "image should have channel axis"

	nX = img.shape[0] - patchSize
	nY = img.shape[1] - patchSize
	nChannels = img.shape[2]
	patches = np.zeros((patchSize, patchSize, nChannels, nX*nY), img.dtype)

	#iterate through all patches from img and store in patches
	k = 0
	for i in range(nX):
		for j in range(nY):
			x,X = i, i+patchSize
			y,Y = j, j+patchSize
			patches[:,:,:,k] = img[x:X,y:Y,:]
			k += 1

	return patches

def getMatchingPatch(distances):
	'''
	Given a 1-D array of patch distances, choose matching patch index.
	'''
	d = distances
	m = np.min(d)
	#choose random index such that the distance is within 1.1x of minimum distance
	threshold = 1.1*m
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

if __name__ == "__main__":
	# read in original image using Python Image Library (PIL)
	orig_img = Image.open("pebbles.png")
	print orig_img.mode, orig_img.size # mode = RGB, size = (88,100)

	# extract list of pixels in RGB/grayscale format
	pixels = list(orig_img.getdata())
	pixels_2d = np.array(pixels, np.int32)
	pixels_2d = pixels_2d.reshape((88,-1,3))

	sl = (slice(0,44), slice(0,50), slice(0,3))
	patch = pixels_2d[sl[0], sl[1], sl[2]]

	textureSize = (100 * 2, 88 * 2)
	

	# img now is an np.ndarray with shape (88,100,3)
	# img = np.reshape(img, (100,88,3), order='C')

	# ensure that img is either an RGB or grayscale image
	# assert img.ndim == 3 and (img.shape[2] == 3 or img.shape[2] == 1), img.shape

	# patches has shape (30,30,3,4060 = 58*70) yay!!!
	# patches = mkPatches(img, 30)
	# img = mkTexture((88, 200), patches, overlap=10)
	
	# define textureSize to be whatever we want
	# textureSize = (100, 88)

	# pixels_out = np.reshape(img, (8800,3), order='C')
	# pixels_out = map(lambda x: (x[0],x[1],x[2]), pixels_out)
	# print pixels_out
	# # print len(pixels_out)
	# # print type(pixels_out)
	# img_out = Image.new(orig_img.mode, textureSize)
	# # print img.shape
	# img_out.putdata(pixels_out)
	# img_out.show()






























	

