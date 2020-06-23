# solution.py ---
#
# Filename: solution.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Sun Jan 14 20:32:47 2018 (-0800)
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


import matplotlib.pyplot as plt
import numpy as np

from config import get_config, print_usage, print_config
from utils.cifar10 import load_data
from utils.features import extract_h_histogram, extract_hog
from utils.preprocess import normalize
from utils import loss_func


def compute_feature(input, config):
    if config.feature_type == "hog":
        # HOG features
        output = extract_hog(input)
    elif config.feature_type == "h_histogram":
        # Hue Histogram features
        output = extract_h_histogram(input)
    elif config.feature_type == "rgb":
        # raw RGB features
        output = input.astype(float).reshape(len(input), -1)
    return output


def compute_loss(W, b, x, y, config):
    """Computes the losses for each module."""

    if config.model_type == "linear_svm":
        model_loss = loss_func.linear_svm
    elif config.model_type == "logistic_regression":
        model_loss = loss_func.logistic_regression
    else:
        raise ValueError("Wrong model type {}".format(
            config.model_type))

    loss = model_loss(W, b, x, y)

    return loss


def predict(W, b, x, config):
	#import IPython
	#IPython.embed()
	z = x@W+b
	pred = np.empty(shape=x.shape[0])
	for x in range(0,len(pred)):
		position_val = np.where(z[x] == z[x].max())
		pred[x] = position_val[0][0]
	return pred


def main(config):
    print_config(config)
    """The main function."""

    # ----------------------------------------
    # Load cifar10 train data
    print("Reading training data...")
    data_trva, y_trva = load_data(config.data_dir, "train")

    # ----------------------------------------
    # Load cifar10 test data
    print("Reading test data...")
    data_te, y_te = load_data(config.data_dir, "test")

    # ----------------------------------------
    # Extract features
    print("Extracting Features...")
    x_trva = compute_feature(data_trva, config)
    x_te = compute_feature(data_te, config)

    # ----------------------------------------
    # Load W, b, mean, range
    x_mean = np.load("./data/{}_mean.npy".format(config.feature_type))
    x_range = np.load("./data/{}_range.npy".format(config.feature_type))
    W = np.load("./data/{}_W.npy".format(config.feature_type))
    
    b = np.load("./data/{}_b.npy".format(config.feature_type))
    import IPython
    IPython.embed()
    # Check that the newly computed mean and range is the same as the loaded
    # one
    print("Testing mean and variance")
    _, x_tr_mean, x_tr_range = normalize(x_trva)
    if not np.all(np.isclose(x_tr_mean, x_mean)) or \
       not np.all(np.isclose(x_tr_range, x_range)):
        print("Mean, range computation is not propper!")
    else:
        print("Mean, range is identical")

    # Apply normalization to the data
    x_te_n, _, _ = normalize(x_te, x_mean, x_range)

    # Predict the labels
    pred = predict(W, b, x_te_n, config)

    # Check accuracy (i.e. count how many are equal)
    acc = np.mean(pred == y_te)

    print("Test Accuracy: {}%".format(acc * 100))

    # Check losses (not necessary, helps you understand what's going on)
    loss = np.mean(compute_loss(W, b, x_te_n, y_te, config))
    print("Average Test loss: {}".format(loss))


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)


#
# solution.py ends here
