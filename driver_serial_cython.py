import sys
import os.path
sys.path.append(os.path.join('..', 'util'))

import numpy as np
from PIL import Image
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

from quilting_serial_cython import *

# TODO: look at Liang paper

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

	# generate all sample patches
	patches = makePatches(sample_2d, patchSize)

	# TODO: randomly select initial patch
	initialPatch = np.zeros((patchSize, patchSize, 3), dtype=np.float32)
	initialPatch[:,:,:] = patches[0,:,:,:]

	# define textureSize, tileSize and initialize blank canvas
	textureSize = (width * 2, height * 2)
	overlap = patchSize / 6
	tileSize = patchSize - overlap
	texture = np.zeros((textureSize[1], textureSize[0], 3), dtype=np.float32)
	textureWidth = textureSize[0]
	textureHeight = textureSize[1]

	numPatches = patches.shape[0]

	# paste all patches: paste includes 1) selecting from candidate patches, 2) calculating min error boundary
	# and 3) inserting patches into output texture
	# TODO: think about whether it should return or remain as void function
	# insert(texture, initialPatch, 0, 0, 0)
	
	N = int(math.ceil(textureSize[0]/float(tileSize)))
	M = int(math.ceil(textureSize[1]/float(tileSize)))

	tid = -1

	for rowNo in range(M): # height M
		for colNo in range(N): # width N
			d = np.zeros(patches.shape[0], dtype=np.float32)
			distLeft = np.zeros(patches.shape[0], dtype=np.float32)
			distUp = np.zeros(patches.shape[0], dtype=np.float32)
			distBoth = np.zeros(patches.shape[0], dtype=np.float32)
			# TODO: double?
			distances = np.empty_like(patches, dtype=np.float32)
			pathCostsLeft = np.zeros((tileSize, overlap), dtype=np.int32)
			pathCostsUp = np.zeros((overlap, tileSize), dtype=np.int32)
			# refPatchUp.shape[0], refPatchLeft.shape[1]
			pathMaskBoth = np.zeros((overlap, overlap), dtype=np.int32)

			costMapLeft = np.zeros((tileSize, overlap), dtype=np.float32)
			costMapUp = np.zeros((overlap, tileSize), dtype=np.float32)

			tid += 1

			print "On iteration %i" % tid

			# insert default initial top-left patch
			if tid == 0:
				insert(texture, initialPatch, rowNo, colNo, tid)
				continue

			blockLeft = 1 if colNo>0 else 0
			blockUp = 1 if rowNo>0 else 0
			
			# find reference patchs and calculate overlap distances over all sample patches
			if blockLeft:
				# TODO does using python min affect performance enough?
				refPatchLeft = texture[rowNo*tileSize:min(rowNo*tileSize + patchSize, textureHeight), 
								colNo*tileSize:min(colNo*tileSize + overlap, textureWidth), :]
				overlapDistances(refPatchLeft, patches, distances, distLeft)
				# reference or actual copy if d = distLeft + distUp - distBoth didn't work
				d = distLeft

			if blockUp:
				refPatchUp = texture[rowNo*tileSize:min(rowNo*tileSize + overlap, textureHeight), 
								colNo*tileSize:min(colNo*tileSize + patchSize, textureWidth), :]
				overlapDistances(refPatchUp, patches, distances, distUp)
				d = distUp

			if blockLeft and blockUp:
				refPatchBoth = texture[rowNo*tileSize:min(rowNo*tileSize + overlap, textureHeight), 
								colNo*tileSize:min(colNo*tileSize + overlap, textureWidth), :]
				overlapDistances(refPatchBoth, patches, distances, distBoth)
				# cythonized version of: d = distLeft + distUp - distBoth
				for i in range(numPatches):
					d[i] = distLeft[i] + distUp[i] - distBoth[i]

			# finds appropriate random patch
			chosenIdx = getMatchingPatch(d, 1.1, tid)
			chosenPatch = patches[chosenIdx, :, :, :]

			# determines minimum cut boundary and overlays onto chosen patch
			if blockLeft:
				makeCostMap(refPatchLeft, chosenPatch[:refPatchLeft.shape[0], :overlap, :], costMapLeft, tid)
				cheapVertCut(costMapLeft, pathCostsLeft, tid)
				# TODO: fix?
				combineRefAndChosen(pathCostsLeft, refPatchLeft, chosenPatch, 0, overlap, tid)

			if blockUp:
				# chosenSize = min(colNo*tileSize + patchSize, textureWidth) - colNo*tileSize
				# TODO: stupid solution; find better one
				makeCostMap(refPatchUp, chosenPatch[:overlap, :refPatchUp.shape[1], :], costMapUp, tid)
				cheapHorizCut(costMapUp, pathCostsUp, tid)
				combineRefAndChosen(pathCostsUp, refPatchUp, chosenPatch, 1, overlap, tid)

			if blockLeft and blockUp:

				for i in range(overlap):
					for j in range(overlap):
						# bitwise or operation
						pathMaskBoth[i,j] = 1 - ((1-pathCostsUp[i,j]) * (1-pathCostsLeft[i,j]))

				pathCostsLeft[:overlap,:] = pathMaskBoth
				pathCostsUp[:,:overlap] = pathMaskBoth

				combineRefAndChosen(pathCostsLeft, refPatchLeft, chosenPatch, 0, overlap, tid)
				combineRefAndChosen(pathCostsUp, refPatchUp, chosenPatch, 1, overlap, tid)

			insert(texture, chosenPatch, rowNo*tileSize, colNo*tileSize, tid)


	# convert texture into flattened array pixels_out for exporting as PNG
	pixels_out = np.reshape(texture, (textureSize[0] * textureSize[1], 3), order='C')
	pixels_out = map(lambda x: (x[0],x[1],x[2]), pixels_out)
	img_out = Image.new(orig_img.mode, textureSize)
	img_out.putdata(pixels_out)
	img_out.show()
	print "donedonedone!\n"
