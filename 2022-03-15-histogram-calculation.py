import cv2
import time
import utils
import numpy as np
from numba import jit
from pathlib import Path

examples_folder = Path("../project configuration/OpenCV 4 - C++/example_images")


def our_hist(img):
    freqs = np.zeros((255,))
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            freqs[img[y, x]] += 1
    return freqs


@jit(nopython=True)
def our_hist_numba(img):
    freqs = np.zeros((255,))
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            freqs[img[y, x]] += 1
    return freqs


def main():
    img = cv2.imread(
        str(examples_folder/'lowcontrast.png'), cv2.IMREAD_GRAYSCALE
    )

    # Numpy version
    start = time.time()
    freqs, values = np.histogram(img.flatten(), bins=255)
    print(f'Numpy version time: {(time.time()-start):.3f}')

    # Our version python
    start = time.time()
    _ = our_hist(img)
    print(f'Our version time: {(time.time()-start):.3f}')

    # Our version numba
    start = time.time()
    _ = our_hist_numba(img)
    print(f'Numba version first time: {(time.time()-start):.3f}')

    # Our version numba
    start = time.time()
    _ = our_hist_numba(img)
    print(f'Numba version second time: {(time.time()-start)}')

    utils.hist_plot(values, freqs)


if __name__ == '__main__':
    main()
