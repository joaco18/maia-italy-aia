import cv2
import utils
import time
import numpy as np
from numba import jit


@jit(nopython=True)
def our_hist_numba(img: np.ndarray):
    # Histogram
    freqs = np.zeros((256,))
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            freqs[img[y, x]] += 1
    return freqs


@jit(nopython=True)
def our_cdf_numba(img: np.ndarray):
    # Histogram
    freqs = our_hist_numba(img)
    # Histogram normalization
    freqs = freqs / img.size
    return np.cumsum(freqs)


def our_cdf(img: np.ndarray):
    # Histogram
    freqs = np.zeros((256,))
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            freqs[img[y, x]] += 1
    # Histogram normalization
    freqs = freqs / img.size
    return np.cumsum(freqs)


def equalize_hist(img: np.ndarray):
    """
    Equalization of 8-bit grayscale image
    Args:
        img (np.ndarray): one channel image to equalize.
    """
    assert len(img.shape) == 2, \
        "In equalize_hist: Multichannel images not supported"

    # CDF time performace:
    # start = time.time()
    # cdf = our_cdf_numba(img)
    # print(f'Numba func: {time.time()-start}')

    # start = time.time()
    # cdf = our_cdf(img)
    # print(f'No-numba func: {time.time()-start}')

    # CDF
    cdf = our_cdf_numba(img)
    # Hist equalization LUT
    cdf = (cdf * 255).astype(int)
    return cdf[img]


def main():
    img = cv2.imread(
        str(utils.EXAMPLES_DIR/'lightning_gray.jpg'),
        cv2.IMREAD_GRAYSCALE
    )
    freqs = our_hist_numba(img)
    intensities = np.arange(0, 257)
    utils.cv2_imshow('Original Image', img, cmap='gray')
    utils.hist_plot(intensities, freqs, 'Original Image Histogram')

    res = img.copy()
    cv2.normalize(img, res, 0, 255, cv2.NORM_MINMAX)
    freqs = our_hist_numba(res)
    utils.cv2_imshow('Normalized Image', res, cmap='gray')
    utils.hist_plot(intensities, freqs, 'Normalized Image Histogram')

    clahe_filter = cv2.createCLAHE(10)
    res = clahe_filter.apply(img)
    utils.cv2_imshow('Clahe Enhaced Image', res, cmap='gray')
    utils.hist_plot(intensities, freqs, 'Clahe Enhaced Image Histogram')

    res = equalize_hist(img)
    freqs = our_hist_numba(res)
    utils.cv2_imshow('Equalized Image', res, cmap='gray')
    utils.hist_plot(intensities, freqs, 'Equalized Image Histogram')


if __name__ == '__main__':
    main()
