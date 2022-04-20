import cv2
import utils
import numpy as np
from pathlib import Path

examples_folder = Path("../project configuration/OpenCV 4 - C++/example_images")


def main():
    img = cv2.imread(str(examples_folder/'lowcontrast.png'), cv2.IMREAD_GRAYSCALE)
    assert img is not None, 'Couldn\'t load the image'
    utils.cv2_imshow("Original image", img, 'gray')
    freqs, vals = np.histogram(img.flatten(), 255)
    utils.hist_plot(vals, freqs)

    img = (img - img.min()) / (img.max() - img.min())
    # cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    utils.cv2_imshow("Normalized image", img, 'gray')
    freqs, vals = np.histogram(img.flatten(), 255)
    utils.hist_plot(vals, freqs)


if __name__ == '__main__':
    main()
