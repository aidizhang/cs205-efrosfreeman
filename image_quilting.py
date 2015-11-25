#!/usr/bin/python

import numpy as np
import math
import sys
import os
import png
import itertools

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

if __name__ == "__main__":
	#load image
	reader = png.Reader("pebbles.png")
	x, y, pixels, meta = reader.asDirect()
	img = png.Image(pixels,meta)
	print img
	print img.shape
	# img = np.vstack(itertools.imap(np.uint16, pngdata))
	#make sure we have a grayscale/RGB image
	# print img.shape, type(img)

