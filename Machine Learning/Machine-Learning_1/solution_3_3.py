import sys

import matplotlib.pyplot as plt

try:
    from csc_486b_assignments.solution.assignment1 import utils
except ModuleNotFoundError:
    import utils


if __name__ == "__main__":
    k1 = int(sys.argv[1])
    k2 = int(sys.argv[2])

    img_inv = utils.read_image_from_h5("output.h5")
    res = utils.find_difference_of_gaussian_blur(img_inv, k1, k2)

    plt.figure()
    plt.imshow(res, cmap="gray")
    plt.show()

    utils.save_image_to_h5(res, "filtered.h5")
