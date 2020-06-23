import matplotlib.pyplot as plt

try:
    from csc_486b_assignments.solution.assignment1 import utils
except ModuleNotFoundError:
    import utils


if __name__ == "__main__":

    filtered = utils.read_image_from_h5("filtered.h5")

    thresholded = utils.keep_top_percentile(filtered, 5)

    plt.figure()
    plt.imshow(thresholded, cmap="gray")
    plt.show()
