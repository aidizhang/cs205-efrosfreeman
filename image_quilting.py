#!/usr/bin/python

import numpy as np
import math
import sys
import os
import png
import itertools
from pylab import *
import matplotlib.pyplot as plt

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
	#note that img should have channel axis, so (x,y,channel)
	width = img.shape[0] - patchSize
	height = img.shape[1] - patchSize
	nChannels = img.shape[2]
	patches = np.zeros((patchSize, patchSize, nChannels, width*height), img.dtype)

	#iterate through all patches from img and store in patches
	k = 0
	for i in width:
		for j in height:
			x1,x2 = i, i+patchSize
			y1,y2 = j, j+patchSize
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
	nChannels = patches.shape[2]
	tileSize = patchSize - overlap
	texture = np.zeros((textureSize[0], textureSize[1], nChannels), dtype=np.float32)

	nPatches = patches.shape[2]

	k = -1
	width, height = int(math.ceil(textureSize[0]/float(tileSize))), int(math.ceil(textureSize[0]/float(tileSize)))
	for i in range(width):
		for j in range(height):
			k += 1

			#use random patch everytime
			texture[0:patchSize, 0:patchSize, :] = patches[:,:,:,np.random.randint(0,nPatches)]

	return texture

if __name__ == "__main__":
	img = Image.open("pebbles.png")
	pixels = img.load()
	#convert Image instance to numpy array for manipulation
	img = np.array(img)
	#print img.shape (88,100,3)
	plt.imshow(img)
	#img.save("pebbles_generate.png", "png")

	#load image
	# fname = "pebbles.png"
	# reader = png.Reader(fname)
	# w,h,pixels,metadata = reader.read_flat() #or reader.read(), asDirect()
	# output = open("pebbles_new", "wb")
	# writer = png.Writer(w,h,**metadata)
	# writer.write_array(output, pixels)
	# output.close()
	# img = np.vstack(itertools.imap(np.uint8, pixels))
	#print img.shape - shape (88,300)
	# print img[20,:]
	# write_png("pebbles_generate.png", img)


	#patchSize = 30
	# patches = mkPatches(img, 30)
	
	#textureSize = (300,150)
	# img = mkTexture((300,150), patches, overlap=10)

	#write out image to file
	# outFname = os.path.splitext(fname)[0]+"_generated.png"
	# write_png('pebbles_generate.png', img)













