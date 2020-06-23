# features.py ---
#
# Filename: features.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Sun Jan 14 21:06:57 2018 (-0800)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
# Copyright (C), Visual Computing Group @ University of Victoria.

# Code:

import numpy as np
from skimage.color import rgb2hsv,hsv2rgb
from skimage.feature import hog
from skimage.color import rgb2gray


def extract_h_histogram(data):


	#import IPython
	#IPython.embed()
	h_hist = np.empty(shape = (data.shape[0],16))
	for x in range (0,len(data)):
		hsv_image = rgb2hsv(data[x])
		hue_image = hsv_image[:,:,0]
		h_hist_master,bin_edges = np.histogram(hue_image,bins=16,range=(0,1))
		h_hist[x] = h_hist_master		
		assert h_hist.shape == (data.shape[0], 16)		
		
		
	
	return h_hist

def extract_hog(data):
	print("Extracting HOG... ")
	hog_feat = np.empty(shape=(data.shape[0],324))
	for x in range(0,len(data)):
	
		assert hog_feat.shape == (data.shape[0], 324)
		new_image = data[x].mean(axis=-1)
		hog_feat_master =  hog(new_image)	
		hog_feat[x] = hog_feat_master
	return hog_feat

#
# features.py ends here
