# preprocess.py ---
#
# Filename: preprocess.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Mon Jan 15 10:10:03 2018 (-0800)
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


def normalize(data, data_mean=None, data_range=None):
	data_new = data.astype(float)
	if data_mean is None:
		data_mean = np.mean(data_new,axis=0,keepdims=True)
	data_n = data_new - data_mean
	
	if data_range is None:
		data_range = np.std(data_new,axis=0,keepdims=True)
	data_n /= data_range	
	return data_n, data_mean, data_range


#
# preprocess.py ends here
