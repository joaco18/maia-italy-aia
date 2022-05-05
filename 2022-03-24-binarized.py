import cv2
import utils
import numpy as np


def main():
    input_img = cv2.imread(
        str(utils.EXAMPLES_DIR / 'tools.png'),
        cv2.IMREAD_GRAYSCALE
    )
    assert input_img is not None, 'Image cannot be loaded'
    utils.cv2_imshow('Original image', input_img, 'gray')
    val = np.arange(256)
    freq = utils.our_hist_numba(input_img)
    utils.hist_plot(val, freq, 'Original image histogram')

    # Otsu
    T, binarized_img = cv2.threshold(
        input_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f'Otsu threshold: {T}')
    utils.cv2_imshow('Otsu image', binarized_img, 'gray')

    # Triangle
    T = utils.get_triangle_auto_threshold(freq)
    print(f'Triangular threshold: {T}')
    _, binarized_img = cv2.threshold(input_img, T, 255, cv2.THRESH_BINARY)
    utils.cv2_imshow('Triangular image', binarized_img, 'gray')


if __name__ == "__main__":
    main()
