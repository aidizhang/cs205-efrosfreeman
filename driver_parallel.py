import sys
import os.path
sys.path.append(os.path.join('..', 'util'))

import numpy as np
from PIL import Image
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

import threading

from quilting_parallel import *

# TODO: look at Liang paper

def schedulePastePatch(texture, patches, initialPatch, metadata, tid, num_threads):
	numRows, numCols = metadata['numRows'], metadata['numCols']

	# for each patch, make sure that left and up patches (if any) are done
	if tid == 0:
		sys.exit(0)

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

	# do work if ready
	pastePatch()

	# mark patch as done!
	events[tid].set()


def parallelPastePatch(texture, patches, initialPatch, metadata):
	# keep track of finished patches in 2D events array
	numThreads = metadata['numCols'] * metadata['numRows']
	events = [threading.Event() for i in range(numThreads)]

	# spawn threads
	threads = []
	for tid in range(num_threads):
		threads.append(threading.Thread(target=schedulePastePatch, 
										args=(texture, patches, initialPatch, metadata, tid, numThreads)))
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

	# create metadata for patch synthesis
	metadata = {'textureSize':textureSize, 
				'overlap':overlap,
				'tileSize':tileSize,
				'numCols':N,
				'numRows':M}

	k = -1

	# TODO #2 - brunt of parallelism, use condition variables
	for i in range(M): # height M
		for j in range(N): # width N
			k += 1
			# TODO counter
			print "On iteration %i" % k
			# insert default initial top-left patch
			if k == 0:
				insert(texture, initialPatch, i, j)
				continue

			blockLeft = j>0
			blockUp = i>0

			# allocate memory for overlap, distances, and results
			# TODO double?
			distances = np.empty_like(patches, dtype=np.float32)
			# find reference patchs and calculate overlap distances over all sample patches
			if blockLeft:
				refPatchLeft = texture[i*tileSize:min(i*tileSize + patchSize, textureSize[1]), 
								j*tileSize:min(j*tileSize + overlap, textureSize[0]), :]
				distLeft = np.zeros(patches.shape[0], dtype=np.float32)
				overlapDistances(refPatchLeft, patches, distances, distLeft)
				d = distLeft

			if blockUp:
				refPatchUp = texture[i*tileSize:min(i*tileSize + overlap, textureSize[1]), 
								j*tileSize:min(j*tileSize + patchSize, textureSize[0]), :]
				distUp = np.zeros(patches.shape[0], dtype=np.float32)
				overlapDistances(refPatchUp, patches, distances, distUp)
				d = distUp

			if blockLeft and blockUp:
				refPatchBoth = texture[i*tileSize:min(i*tileSize + overlap, textureSize[1]), 
								j*tileSize:min(j*tileSize + overlap, textureSize[0]), :]
				distBoth = np.zeros(patches.shape[0], dtype=np.float32)
				overlapDistances(refPatchBoth, patches, distances, distBoth)
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
	print "donedonedone!\n"
