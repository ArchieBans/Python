import matplotlib.pyplot as plt
import utils

if __name__ == "__main__":

    img = utils.read_image("input.jpg")
    img_inv = utils.invert_image(img)
    utils.save_image_to_h5(img_inv, "output.h5")

    plt.figure()
    plt.imshow(img)
    plt.figure()
    plt.imshow(img_inv)
    plt.show()
