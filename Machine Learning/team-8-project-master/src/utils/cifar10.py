# cifar10.py ---
#
# Filename: cifar10.py
# Description:
# Author: Archit Kumar
# Maintainer:
# Created: Sun Jan 14 20:44:24 2018 (-0800)
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

import os

import numpy as np

from utils.external import unpickle


def load_data(data_dir, data_type):
    """Function to load data from CIFAR10.

    Parameters
    ----------
    data_dir : string
        Absolute path to the directory containing the extracted CIFAR10 files.

    data_type : string
        Either "train" or "test", which loads the entire train/test data in
        concatenated form.

    Returns
    -------
    data : ndarray (uint8)
        Data from the CIFAR10 dataset corresponding to the train/test
        split. The datata should be in NHWC format.

    labels : ndarray (int)
        Labels for each data. Integers ranging between 0 and 9.

    """

    if data_type == "train":
        data = []
        label = []
        for _i in range(4):
            file_name = os.path.join(data_dir, "data_batch_{}".format(_i + 1))
            cur_dict = unpickle(file_name)
            data += [
                np.array(cur_dict[b"data"])
            ]
            label += [
                np.array(cur_dict[b"labels"])
            ]
        # Concat them
        data = np.concatenate(data)
        label = np.concatenate(label)

    elif data_type == "valid":
        # We'll use the 5th batch as our validation data. Note that this is not
        # the best way to do this. One strategy I've seen is to use this to
        # figure out the loss value you should aim to train for, and then stop
        # at that point, using the entire dataset. However, for simplicity,
        # we'll use this simple strategy.
        data = []
        label = []
        cur_dict = unpickle(os.path.join(data_dir, "data_batch_5"))
        data = np.array(cur_dict[b"data"])
        label = np.array(cur_dict[b"labels"])

    elif data_type == "test":
        data = []
        label = []
        cur_dict = unpickle(os.path.join(data_dir, "test_batch"))
        data = np.array(cur_dict[b"data"])
        label = np.array(cur_dict[b"labels"])

    else:
        raise ValueError("Wrong data type {}".format(data_type))

    # Turn data into (NxCxHxW) format, so that we can easily process it, where
    # N=number of images, H=height, W=widht, C=channels. Note that this
    # is the default format for PyTorch.
    data = np.reshape(data, (-1, 3, 32, 32))

    return data, label


#
# cifar10.py ends here
